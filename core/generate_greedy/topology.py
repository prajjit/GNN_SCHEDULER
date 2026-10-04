from math import ceil
import random

import networkx as nx
import scipy.constants
import numpy as np
import pandas as pd
import pickle

# from schedule import Schedule
from visualization import TopologyVisualizer


def gen_connected_with_retry(graph_generator, max_attempts=10, **args):
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        G = graph_generator(**args)
        if len(G.nodes) == 0 or nx.is_connected(G):
            G.graph['on_attempt'] = attempt # Save this just in case
            return G
    raise Exception(f"Failed to generate a connected graph "
                    f"in {max_attempts} tries.")


class TopologyGraph(nx.Graph):
    """Generic network topology graph

    This class represents and manipulates an heterogeneous network topology
    that consist of a mesh network of conventional wireless nodes with
    additional battery-free tags associated to (hosted by) some of them.

    This class merely implements most of the general functionality in the
    module but it is not meant to be instantiated directly, except when
    explicitly building a specific topology "by hand". It is far more useful
    and convenient to instantiate one of the following TopologyGraph sub
    classes:

        - ErdosTopologyGraph
        - GridTopologyGraph
        - GeometricTopologyGraph
        - RealisticGeometricTopologyGraph

    All TopologyGraphs have a few general topology properties accessible
    through the graph dictionary (e.g.: <TopologyGraph>.graph['n_nodes']).
    These always include: 
        - n_nodes: The number of regular nodes
        - n_tags: The number of tags
        - topology: The name/type of the underlying topology (default: generic)
    TopologyGraph sub classes may add other properties.

    All nodes have a 'type' property. For regular nodes the type is 
    'active'. Passive tags have type 'tag'. Active nodes have the property
    'n_tags' while tags have additional property 'host'.
    
    All edges have a 'weight' property with value DEFAULT_EDGE_STRENGTH=1 by
    default. Edges between tags and their host have a weight of
    HOST_STRENGTH=400 by default.

    Attributes:
        active_nodes: The list of active or regular nodes
        hosts: The subset of active nodes that have associated tags
        tags: The list of tags in the topology
        multi_tag: Allow multiple tags per host
    """
    _DEFAULT_EDGE_STRENGTH = 1
    _HOST_STRENGTH = 400
    _NODE_LABEL_PREFIX = 'A'
    _TAG_LABEL_PREFIX = 'T'

    def __init__(self, incoming_graph_data=None, multi_tag=True, **attr):
        # if incoming_graph_data is not None:
            # nx.set_node_attributes(incoming_graph_data, name='type',   values='active')
            # nx.set_node_attributes(incoming_graph_data, name='n_tags', values=0)
            # print(incoming_graph_data.nodes(data=True))
        super().__init__(incoming_graph_data, **attr)

        self.graph.setdefault('n_nodes', len(self.nodes))
        self.graph.setdefault('n_tags', 0)
        self.graph.setdefault('topology', 'generic')
        if self.graph['n_nodes'] == 0 and self.graph['n_tags'] > 0:
            raise Exception(f"Cannot add {self.graph['n_tags']} tag(s) to "
                            f"topology with 0 active nodes.")

        for f, t in self.edges:
            if not 'weight' in self[f][t]:
                self[f][t]['weight'] = self._DEFAULT_EDGE_STRENGTH

        nx.relabel_nodes(self, 
                         dict([(n, self.node_label_format(n)) for n in self.nodes()]),
                         copy=False)
        nx.set_node_attributes(self, name='type',   values='active')
        nx.set_node_attributes(self, name='n_tags', values=0)

        self.active_nodes = list(self.nodes)
        self.hosts = set()
        self.tags  = list()
        tags_to_add = self.graph['n_tags']
        # Temporarily set to zero... add_random_tags will restore it
        self.graph['n_tags'] = 0
        self.add_random_tags(tags_to_add, multi_tag)
        self._compute_attributes()

    @property
    def assignments(self):
        """The tag-to-host assignment as a dictionary
        """
        return {t:self.nodes[t]['host'] for t in self.tags}

    def add_random_tags(self, n_tags, multi_tag=True):
        """Add tags next to random nodes
        
        Add <n_tags> new tags next to randomly chosen active node 'hosts'. The
        hosts are chosen so that there will always be at least one potential
        carrier generator available. 

        Args:
            n_tags: The number of tags to add
            multi_tag: Allow multiple tags per host
        """
        # Make sure we only add tags to nodes with a potential
        # carrier generator as neighbor
        potential_hosts = [n for n in self.active_nodes if
                self.active_degree(n)>0 and (multi_tag or
                    self.nodes[n]['n_tags']==0)] 
        if multi_tag:
            hosts = random.choices(potential_hosts, k = n_tags)
        else:
            hosts = random.sample(potential_hosts, k = min(n_tags, len(potential_hosts)))
        current_n_tags = len(self.tags)
        new_tags = map(self.tag_label_format, 
                range(current_n_tags, current_n_tags+len(hosts)))
        self.add_tags(zip(new_tags, hosts))

    def active_neighbors(self, node):
        """The set of neighbors of a node that are also active nodes
        """
        return set(self.neighbors(node)) & set(self.active_nodes)

    def active_degree(self, node):
        """The number of active node neighbors a node has
        """
        return len(self.active_neighbors(node))

    def copy(self):
        ret = super().copy()
        ret._compute_attributes()
        return ret

    @staticmethod
    def node_label_format(n):
        return f'{TopologyGraph._NODE_LABEL_PREFIX}{n}'

    @staticmethod
    def tag_label_format(n):
        return f'{TopologyGraph._TAG_LABEL_PREFIX}{n}'

    @staticmethod
    def node_id_from_label(label):
        return int(label[len(TopologyGraph._NODE_LABEL_PREFIX):])

    @staticmethod
    def tag_id_from_label(label):
        return int(label[len(TopologyGraph._TAG_LABEL_PREFIX):])

    def add_tags(self, tag_assignment):
        """Add tags to graph according to tag-to-host assignment

        Adds tags 'next' to the corresponding hosts as indicated.

        Args:
            tag_assignment: A dictionary {t:h, ...} or a list of two-tuples 
            (t, h) indicating that tag <t> should be hosted by node <h>.  
        """
        tag_assignment = dict(tag_assignment)
        for tag, node in tag_assignment.items():
            if node not in self.nodes():
                raise Exception(f'Adding tag {tag} to missing node {node}.')
            self.add_node(tag, **{'type': 'tag',
                'host': node, 'pos': (random.uniform(0.99, 1.09)*self.nodes[node]['pos'][0], random.uniform(0.99, 1.09)*self.nodes[node]['pos'][1])})
            self.add_edge(node, tag,
                    **{'weight': self._HOST_STRENGTH})
            self.nodes[node]['n_tags'] += 1
            self.graph['n_tags'] += 1
            self.hosts.add(node)
            self.tags.append(tag)
            for n in self.neighbors(node):
                if n != tag:
                    self.add_edge(n, tag, **{'weight': self.edges[node,n]['weight']})

    def get_adjacency_matrix_string(self):
        """Get a string representing the adjacency matrix of the topology
        
        This is a way to represent a graph in MiniZinc format as a 
        2D array.
        
        Returns:
            A string formatted to represent the graph in MiniZinc format  
        """
        S = '['
        if len(self.nodes) > 0:
            a = nx.adjacency_matrix(self, nodelist=sorted(self.nodes(), key=self.node_id_from_label))
            for i, _ in enumerate(sorted(self.nodes(), key=self.node_id_from_label)):
                if i > 0:
                    S += ' '
                S += '| '
                for j, _ in enumerate(sorted(self.nodes(), key=self.node_id_from_label)):
                    S += '{},'.format(a[i,j])
                S += ' \n'
        S += ' |];'
        return S

    def get_problem_string(self, n_slots=None):
        """This creates a formatted MiniZinc input
        
        This creates a formatted MiniZinc input corresponding to this topology.
        
        Args:
            n_slots: Number of slots in the slotframe. If `None`, use one slot
            per tag, which is the worst case.
            
         Returns:
            S: A string containing the MiniZinc formatted problem string
        """
        tag_ids_list = sorted(self.tags, key=self.tag_id_from_label)
        node_ids_list = sorted(self.active_nodes, key=self.node_id_from_label)
        node_ids_list.extend(tag_ids_list)
        if n_slots is None:
            n_slots = self.graph['n_tags']
        S  = f'nSlots = {n_slots};\n'
        S += 'NODES = {'
        S += ', '.join(node_ids_list)
        S += '};\n'
        S += 'TAGS = {'
        S += ', '.join(tag_ids_list)
        S += '};\n'
        S += 'HOSTS = array1d(TAGS, ['
        for n in tag_ids_list:
            S += '{}, '.format(self.nodes[n]['host'])
        S += ']);\n\n'
        S += 'connectivity = ' + self.get_adjacency_matrix_string()
        return S
            
    def clear_irrelevant_nodes(self):
        """Remove all nodes that are not connected to any tag

        Nodes that are neither neighbors to any host or hosts themselves are
        trivially OFF during the whole tag schedule. This remove those
        irrelevant nodes from the topology.

        Returns:
            A copy of the topology with all the 'irrelevant' nodes removed.
            A list of the nodes that were removed
        """
        non_relevant = list()
        for node in self.active_nodes:
            is_relevant = False
            for neigh in self.neighbors(node):
                if self.nodes[neigh]['type'] == 'tag':
                    is_relevant = True
                    break
            if not is_relevant:
                non_relevant.append(node)
        new_graph = self.copy()
        new_graph.remove_active_nodes(non_relevant)
        return new_graph, non_relevant

    def conflict_graph(self):
        """Compute the conflict graph of this topology

        The conflict graph contains the same (active) nodes as the original
        topology but edges between any given pair of nodes indicate that they
        would create interference on a neighboring tag if they were to generate
        a carrier simultaneously. 

        Returns:
            A new topology graph of the same type, with the same active nodes
            but without tags and with edges indicating potential conflicts as
            described above.
        """
        H = self.copy()
        H.remove_tags(H.tags)
        H.remove_edges_from(H.edges)
        for i in self.hosts: #only relevant for nodes with tags
            active_neighbors = list(self.active_neighbors(i))
            for l in active_neighbors[:-1]:
                for m in active_neighbors[1:]:
                    H.add_edge(l, m)
        return H

    def _compute_attributes(self):
        self.tags = [n for n in self.nodes if 'type' in self.nodes[n] and 
                self.nodes[n]['type'] == 'tag']
        self.active_nodes = [n for n in self.nodes 
                if 'type' in self.nodes[n] and self.nodes[n]['type'] == 'active']
        self.hosts = set([self.nodes[t]['host'] for t in self.tags])
        self.graph['n_nodes'] = len(self.active_nodes)
        self.graph['n_tags'] = len(self.tags)

    def connected_components(self):
        """Yield each connected component in the topology as individual TopologyGraphs
        """
        for c in nx.connected_components(self):
            c = self.subgraph(c).copy()
            c._compute_attributes()
            yield c

    def remove_tags(self, tags):
        """Remove a list of tags from the topology

        Makes sure to keep all the attributes updated.
        """
        # Make a copy because often tags == self.tags
        for t in tags.copy():
            if t in self:
                self.graph['n_tags'] -= 1
                self.tags.remove(t)
                host = self.nodes[t]['host']
                self.nodes[host]['n_tags'] -= 1
                if self.nodes[host]['n_tags'] == 0:
                    self.hosts.remove(host)
                self.remove_node(t)

    def get_hosted_tags(self, node):
        """Get the list of tags hosted by a node
        """
        if node not in self:
            raise Exception(f"Node {node} is not in the topology")
        return [t for t in self.tags if self.nodes[t]['host'] == node]

    def add_nodes_from(self, nodes_for_adding, **attr):
        super().add_nodes_from(nodes_for_adding, **attr)
        self._compute_attributes()

    def remove_active_nodes(self, nodes):
        """Remove a list of active nodes from the topology

        Makes sure to keep all the attributes updated. The node's hosted tags,
        if any, will be removed too.
        """
        for n in nodes:
            if n in self:
                self.remove_tags(self.get_hosted_tags(n))
                self.graph['n_nodes'] -= 1
                self.active_nodes.remove(n)
                if n in self.hosts:
                    self.host.remove(n)
                self.remove_node(n)

    def position_nodes(self, iterations=100, k=0.2):
        positions = {n:self.nodes[n]['pos'] \
                for n in self.nodes if 'pos' in self.nodes[n]
                }
        return nx.layout.spring_layout(self, 
                pos=positions if len(positions) > 0 else None,
                fixed=positions.keys() if len(positions)>0 else None,
                iterations=iterations, k=k)

    def show(self, **kwargs):
        """Draw the network topology using networkx's drawing functionality

        Args:
            Any keyword argument is passed directly to the
            `visualization.TopologyVisualizer.show` method. 
        """
        TopologyVisualizer.show(self, **kwargs)

    def filter_edges(self, min_weight=-1, max_weight=1000000):
        remove_edges = [e for e in self.edges if self.edges[e]['weight'] > max_weight]
        remove_edges.extend(
                [e for e in self.edges if self.edges[e]['weight'] < min_weight])
        self.remove_edges_from(remove_edges)

    def save(self, path):
        """Save the topology in Python pickle format"""
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path, clean=False):
        """Read topology from Python pickle format"""
        with open(path, 'rb') as f:
            ret = pickle.load(f)
        #ret =  nx.read_gpickle(path)
        if not isinstance(ret, TopologyGraph):
            ret = TopologyGraph(ret)
        if clean:
            ret.remove_tags(ret.tags)
        return ret

    def make_reshufle_plan(self):
        """Create a random mapping to shuffle all node and tag IDs

        Returns:
            Two dictionaries mapping the current IDs to the shuffled ones.
            The first dictionary is for active nodes the second one for the
            tags. 
        """
        active_nodes_plan = dict(zip(
            self.active_nodes, 
            random.sample(self.active_nodes, len(self.active_nodes))
            ))
        tags_plan = dict(zip(
            self.tags, 
            random.sample(self.tags, len(self.tags))
            ))
        return active_nodes_plan, tags_plan

    def do_reshufle_ids(self, plan=None):
        """Get a copy of topology with shuffled IDs following a plan

        Args:
            plan: An ID shufle plan as returned by make_reshufle_plan
            represented as two dictionaries. If None, shuffle at random.

        Returns:
            A copy of the topology with shuffled IDs
        """
        if plan is None:
            plan = self.make_reshufle_plan()
        whole_plan = dict(**plan[0], **plan[1])
        H = nx.relabel_nodes(self, whole_plan, copy=True)
        for t in plan[1].values():
            H.nodes[t]['host'] = plan[0][H.nodes[t]['host']]
        H._compute_attributes()
        return H


class ErdosTopologyGraph(TopologyGraph):
    """Topology graph based on a random binomial graph

    This builds a TopologyGraph with an underlying active node topology graph
    based on a random Erdos-Renyi or binomial graph. 

    Args:
        n_nodes: The number of active nodes
        n_tags: The number of tags
        p: Probability of edge creation (optional, default=1.)
        multi_tag: Allow multiple tags per host
        max_attempts: Max attempts at obtaining a connected graph. (optional,
            default=10) 
    """
    def __init__(self, n_nodes=0, n_tags=0, p=1, multi_tag=True, max_attempts=10):
        G = gen_connected_with_retry(
                nx.gnp_random_graph, max_attempts=max_attempts,
                **{'n': n_nodes, 'p': p}
                )
        G.graph['topology'] = 'random'
        G.graph['p'] = p
        super().__init__(incoming_graph_data=G, 
                n_nodes=n_nodes, n_tags=n_tags, multi_tag=multi_tag )


class GridTopologyGraph(TopologyGraph):
    """Topology with active nodes located on a uniform rectangular grid.

    Notice that the actual number of active nodes in the topology will be
    ceil(n_nodes/grid_side) * grid_side. In general this will differ from
    n_nodes. We keep the originally requested n_nodes in
    <GridTopologyGraph>.graph['req_n_nodes'].

    Args:
        `n_nodes`: The number of active nodes
        `n_tags`: The number of tags
        `grid_side`: Number of nodes on one side of the grid. (The other side
            will have `ceil(n_nodes/grid_side)`) 
        multi_tag: Allow multiple tags per host
    """
    def __init__(self, n_nodes=0, n_tags=0, grid_side=1, multi_tag=True):
        G = nx.grid_2d_graph(int(grid_side), ceil(n_nodes/grid_side))
        G = nx.convert_node_labels_to_integers(G, label_attribute='pos')
        G.graph['grid_side'] = int(grid_side)
        G.graph['n'] = ceil(n_nodes/grid_side)
        G.graph['topology'] = 'lattice'
        G.graph['req_n_nodes'] = n_nodes
        super().__init__(incoming_graph_data=G, 
                n_nodes=len(G.nodes), n_tags=n_tags, multi_tag=multi_tag)


class GeometricTopologyGraph(TopologyGraph):
    """Topology with active nodes forming a random geometric graph

    Args:
        n_nodes: The number of active nodes
        n_tags: The number of tags
        radius: Edge formation distance threshold (default=1)
        multi_tag: Allow multiple tags per host
        max_attempts: Max attempts at obtaining a connected graph. (optional,
            default=10) 
    """
    def __init__(self, n_nodes=0, n_tags=0, radius=1, multi_tag=True, max_attempts=10):
        if n_nodes > 0:
            G = gen_connected_with_retry(
                    nx.random_geometric_graph, max_attempts=max_attempts,
                    **{'n': n_nodes, 'radius': radius}
                    )
        else:
            G = nx.Graph()
        G.graph['topology'] = 'geometric'
        G.graph['radius'] = radius
        super().__init__(incoming_graph_data=G, 
                n_nodes=n_nodes, n_tags=n_tags, multi_tag=multi_tag)

    def position_nodes(self, iterations=100, k=None):
        return super().position_nodes(iterations=iterations, k=k)


class RealisticGeometricTopologyGraph(TopologyGraph):
    """Topology where the active nodes are positioned 'realistically' 

    Active nodes are distributed randomly in a square of given side length.
    Edges are added according to a threshold distance similar to the
    `GeometricTopologyGraph`. The threshold is scaled to maintain approximately
    constant density. Additionally, the weight of every edge is computed
    according to the free space path loss model. 

    Arguments:
        n_nodes: The number of active nodes
        n_tags: The number of tags
        side: The side length of the square where nodes are distributed
            (default: 100) 
        multi_tag: Allow multiple tags per host
        max_attempts: Max attempts at obtaining a connected graph. (optional,
            default=1000) 
    """ 
    _CC2538_TX_POWER = 7
    _ANT_GAIN = 3
    # Wavelength in m, for frequency G=2.45GHz #
    _FREQUENCY = 2.45e9
    _WAVELENGTH = scipy.constants.c / _FREQUENCY
    _THR_DISTANCE = 50

    # Path loss in dB #
    @staticmethod
    def _path_loss(dist):
        return 20 * np.log10(RealisticGeometricTopologyGraph._WAVELENGTH / (4*np.pi*dist))
    
    @staticmethod
    def _signal_strength(A, B):
        dist = sum(((a-b)**2 for a, b in zip(A, B)))
        return int(np.round(RealisticGeometricTopologyGraph._CC2538_TX_POWER + \
            RealisticGeometricTopologyGraph._ANT_GAIN + \
            RealisticGeometricTopologyGraph._path_loss(dist) + \
            RealisticGeometricTopologyGraph._ANT_GAIN))

    def _geometric_graph_generator(self, n_nodes, side, scale_ratio):
        positions = {n:(np.random.uniform(high=side*np.sqrt(scale_ratio)), 
                        np.random.uniform(high=side*np.sqrt(scale_ratio))) for n in range(n_nodes)}
        return nx.random_geometric_graph(n_nodes, self._THR_DISTANCE, pos=positions)

    def __init__(self, n_nodes=0, n_tags=0, side=100, multi_tag=True,  max_attempts=1000):
        SIDE = int(side)
        scale_ratio = n_nodes/12
        
        if n_nodes > 0:
            G = gen_connected_with_retry(
                    self._geometric_graph_generator, max_attempts=max_attempts,
                    **{'n_nodes': n_nodes, 'side': SIDE, 'scale_ratio': scale_ratio}
                    )
        else:
            G = nx.Graph()
        G.graph['topology'] = 'geometricr'
        G.graph['side'] = SIDE
        G.graph['scale_ratio'] = scale_ratio
        for i, j in G.edges():
            G.edges[i,j]['weight'] = 100+self._signal_strength(G.nodes[i]['pos'],
                                                               G.nodes[j]['pos'])
        super().__init__(incoming_graph_data=G, 
                n_nodes=n_nodes, n_tags=n_tags, multi_tag=multi_tag)

    def position_nodes(self, iterations=100, k=None):
        if k is None:
            k = self.graph['side']/2
        return super().position_nodes(iterations=iterations, k=k)


#TODO Maybe discover these automatically?
TOPOLOGY_CLASSES = {
        'geometric':    GeometricTopologyGraph,
        'geometricr':   RealisticGeometricTopologyGraph,
        'lattice':      GridTopologyGraph,
        'random':       ErdosTopologyGraph
        }
TOPOLOGY_PARAMETERS = {
        'geometric':    ('radius',),
        'geometricr':   ('side',),
        'lattice':      ('grid_side',),
        'random':       ('p',)
        }
TOPOLOGIES = tuple(TOPOLOGY_CLASSES.keys())


class RandomTopologyIterator(object):
    def __init__(self, n_nodes=2, n_tags=1, topology=ErdosTopologyGraph,
            multi_tag=True, parameters={}):
        self._n_nodes = n_nodes
        self._n_tags = n_tags
        self._multi_tag = multi_tag
        self._topology = topology
        self._parameters = parameters

    def __next__(self):
        return self._topology(n_nodes=self._n_nodes, n_tags=self._n_tags,
                multi_tag=self._multi_tag, **self._parameters)


class GraphTopologyIterator(object):
    def __init__(self, graph, n_tags=1, clean=True,
            filter_u=1000000, filter_l=-1):
        self._topology = TopologyGraph.load(graph, clean=clean)
        self._topology.filter_edges(filter_l, filter_u)
        self._n_tags = n_tags

    def __next__(self):
        G = self._topology.copy()
        G.add_random_tags(self._n_tags)
        return G


class PickleTopologyIterator(object):
    def __init__(self, pickle, reshuffle_cnt=0):
        self._graphs = pd.read_pickle(pickle)['graph']
        self._current = 0
        self._reshuffle_cnt = reshuffle_cnt
        self._reshuffle_iterator = None

    def __iter__(self):
        return self

    def __next__(self):
        try:
            G = self._graphs[self._current]
        except KeyError:
            raise StopIteration
        self._current += 1
        return G

    def __len__(self):
        return len(self._graphs)


class ReshuffleTopologyIterator(object):
    def __init__(self, topology, n_rounds=-1, original_first=False):
        self._topology = topology
        self._n_rounds = n_rounds 
        self._run_ctr = -1 if original_first else 0
        self._origial_first = original_first

    def __iter__(self):
        return self

    def __len__(self):
        res = float('inf')
        if self._n_rounds == 0:
            res = 1
        elif self._n_rounds > 0:
            res = self._n_rounds
            res += 1 if self._origial_first else 0
        return res

    def __next__(self):
        self._run_ctr += 1
        # if zero, return only the original topology once
        if self._n_rounds == 0 or \
                (self._origial_first and self._run_ctr == 0):
            if self._run_ctr > 1:
                raise StopIteration
            else:
                return self._topology
        # if positive, return n_rounds of re-shuffled topologies
        elif self._n_rounds > 0:
            if self._run_ctr > self._n_rounds:
                raise StopIteration
        # if n_rounds=-1, infinitely return reshuffled topologies
        return self._topology.do_reshufle_ids()

