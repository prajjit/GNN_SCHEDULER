import tempfile
import subprocess
import os
import re

import networkx as nx
from ortools.sat.python import cp_model

from visualization import ScheduleVisualizer
from topology import TopologyGraph


def _available_cpu_count():
    """Return CPUs available to the process on Linux and Windows."""
    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is not None:
        return len(get_affinity(0))
    return os.cpu_count() or 1


class Schedule(object):
    """This is the object used to represent and manipulate schedules. 

    Conceptually a schedule is an arrangement of 'interrogation slots' that
    indicate the action that every node should take in each of them. Possible
    actions include: Remain idle or 'OFF', generate an unmodulated carrier 'CG'
    or interrogate one of the battery-free tags the node hosts.

    One can index Schedule objects in two ways: 
        
    It is possible to retrieve the complete schedule of a particular node:
    S['A0']

    It is also possible to retrieve the scheduled action of a node in a
    specific interrogation slot: S['A0', 1]

    Args:
        nodes: An iterable containing the active nodes in the schedule
            (default=None) 
        n_slots: The desired number of interrogation slots in the schedule
            (default=0)
    """
    def __init__(self, nodes=None, n_slots=0):
        self._inner = dict()

        if nodes is not None:
            for n in nodes:
                self._inner[n] = ['OFF'] * n_slots

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) > 2:
                raise IndexError("Only one or two indices accepted.")
            return self._inner[key[0]][key[1]]
        else:
            return self._inner[key]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            if len(key) > 2:
                raise IndexError("Only one or two indices accepted.")
            n, s = key
            self._inner[n][s] = value
        else:
            if not isinstance(value, list):
                raise ValueError("Node schedule elements should be lists.")
            self._inner[key] = value

    def __contains__(self, node):
        return node in self._inner

    def __repr__(self):
        return self._inner.__repr__()

    def __len__(self):
        node_list = list(self.active_nodes)
        return len(self[node_list[0]]) if len(node_list) > 0 else 0

    @property
    def carrier_slots(self):
        """The number of carrier slots scheduled"""
        slots = 0
        for k, v in self._inner.items():
            slots += v.count('CG')
        return slots

    @property
    def active_nodes(self):
        """The list of active nodes in the schedule"""
        return self._inner.keys()

    @property
    def tags(self):
        """The set of tags in the schedule"""
        tags = set()
        for sched in self._inner.values():
            for val in sched:
                if val not in ('OFF', 'CG', 'RX', 'RX', 'INT'):
                    tags.add(val)
        return tags

    @staticmethod
    def parse(schedule_string, node_prefix='A'):
        """Parse a schedule as returned by the MiniZinc model
        
        Params:
            schedule_string: A string containing the MiniZinc model result
            node_prefix: The prefix of active node IDs
            
        Returns:
            schedule: A schedule object representing the assignment for every
                node and slot.  None if result is 'Unsatisfiable'.
        """
        if "UNSATISFIABLE" in schedule_string:
            return None, None # , None
        schedule_string = schedule_string.split('\n')
        NODE_LINE_RE = re.compile(fr'(?P<name>{node_prefix}\d+): \t(?P<schedule>((RX|TX|CG|OFF), )+)')
        ADDR_LINE_RE = re.compile(fr'(?P<name>{node_prefix}\d+): \t(?P<address>(({node_prefix}|Z|T)\d+, )+)')
        CARR_SLOTS_LINE_RE = re.compile(r'carrier slots: (?P<slots>\d+)')
        section = 'MODE'
        schedule = Schedule()
        addresses = dict()
        for line in schedule_string[1:-1]:
            if "Address" in line:
                section = 'ADDRESS'
                continue
            elif 'carrier slots' in line:
                m = CARR_SLOTS_LINE_RE.match(line)
                if m is None:
                    continue
                m = m.groupdict()
                continue
            if section == 'MODE':
                m = NODE_LINE_RE.match(line)
                if m is None:
                    continue 
                m = m.groupdict()
                schedule[m['name']] = m['schedule'].split(', ')[:-1]
            elif section == 'ADDRESS':
                m = ADDR_LINE_RE.match(line)
                if m is None:
                    continue
                m = m.groupdict()
                addresses[m['name']] = m['address'].split(', ')[:-1]       
        # print(schedule)
        for node in schedule.active_nodes:
            for i, val in enumerate(schedule[node]):
                if val == 'TX':
                    schedule[node][i] = addresses[node][i]
        return schedule

    @staticmethod
    def parse_dict(dict_str):
        """Parse a schedule from a dictionary style string representation """
        sched_dict = eval(dict_str)
        ret = Schedule()
        for n, s in sched_dict.items():
            ret.add_node(n, s)
        return ret

    @staticmethod
    def parse_array(sched_array):
        """Construct a schedule from a ML-style timeslot label representation"""
        LABEL_TO_SCHED = {0: 'OFF', 1: 'TX', 2: 'CG'}
        ret = Schedule()
        if sched_array is None or len(sched_array) == 0:
            return ret
        for node_id, _ in enumerate(sched_array[0]):
            ret.add_node(
                    TopologyGraph.node_label_format(node_id),
                    [LABEL_TO_SCHED[sched_array[i][node_id]] for i in range(len(sched_array))]
                    )
        return ret

    def compressed(self):
        """Copy of the schedule with empty slots removed"""
        empty_timeslots = list()
        for i in range(len(self)):
            should_keep = False
            for n in self.active_nodes:
                if self[n][i] != 'OFF':
                    should_keep = True
                    break;
            if not should_keep:
                empty_timeslots.append(i)
        ret = Schedule(self.active_nodes, 0)
        for node in self.active_nodes:
            for i in range(len(self)):
                if i not in empty_timeslots:
                    ret[node].append(self[node][i])
        return ret

    def add_node(self, node_id, schedule=None):
        """Add a node to the schedule

        Add a new node to the schedule optionally setting the schedule for the
        added node. Otherwise the node is added with an all 'OFF' schedule.

        Args:
            node_id: The name of the node to add
            schedule: a list of the actions for the new node to take. If None
                the node will be OFF in all slots. (Default: None)
        """
        if schedule is None:
            schedule = ['OFF'] * len(self)
        schedule.extend(['OFF'] * (len(self) - len(schedule)))
        for node in self.active_nodes:
            self[node].extend( ['OFF'] * (len(schedule)-len(self[node])))
        self[node_id] = schedule

    def copy(self):
        ret = Schedule(self.active_nodes)
        for node in self.active_nodes:
            ret[node] = list(self[node])
        return ret

    def extend(self, schedule):
        """Extend the schedule with the nodes and actions from another schedule

        All nodes from the given schedule will be incorporated and their
        original schedules will be carried over

        """
        for node in schedule.active_nodes:
            self.add_node(node, schedule[node])

    def append(self, schedule):
        """Append the given schedule to the end of this one

        Nodes present in the given schedule but not in this one are ignored.

        """
        for node in self.active_nodes:
            if node in schedule:
                self[node].extend(schedule[node])
            else:
                self[node].extend(['OFF'] * len(schedule))

    def show(self, defer=False, regular_slots=0, style=None, legend=False,
            **args): 
        ScheduleVisualizer.show(self, defer=defer, regular_slots=regular_slots,
                style=style, legend=legend, **args)

    def get_mnz_objectives(self):
        ret = {
                'num_cg_slots': self.carrier_slots,
                'sum_int_slots': 0,
                'sum_cg_nodes': 0,
                'tag_slots': [0] * len(self.tags),
                'tag_tag_slots': [0] * len(self.tags),
                'sum_cg_slots': 0,
                'total': 0,
                }
        active_nodes_list = sorted(list(self.active_nodes))
        tags_list = sorted(list(self.tags))
        for n in self.active_nodes:
            for s, task in enumerate(self[n]):
                if task == 'CG':
                    ret['sum_cg_slots'] += s+1
                    # ret['sum_cg_nodes'] += (TopologyGraph.node_id_from_label(n)+1)*(s+1)
                    ret['sum_cg_nodes'] += active_nodes_list.index(n)+1
                elif task[0] == 'T':
                    tag_idx = tags_list.index(self[n][s])
                    ret['tag_slots'][tag_idx] = s+1
                    ret['tag_tag_slots'][tag_idx] = (s+1) * (tag_idx + 1)
        F4 = len(self.tags) * len(self.active_nodes) + 1
        N = len(self.tags)
        F2 = (N+1)**N
        F3 = N * (N+1) * (2*N+1)
        # ret['sum_int_slots'] = sum(ret['tag_tag_slots'])
        ret['sum_int_slots'] = sum([ret['tag_slots'][t] * (N+1)**(N-t-1) for t in range(N)])
        ret['total'] = ret['sum_cg_nodes']
        ret['total'] += F4 * ret['sum_int_slots']
        ret['total'] += F4*F2*ret['num_cg_slots']
        return ret


def compute_schedule_sequential(topology, cg_threshold=30, **args):
    """Compute the sequential schedule of a topology

    This is the most basic, "naive" scheduler that allocates one dedicated time
    slot and carrier generator to every tag. This is the baseline, worst case
    schedule. Because we have a dedicate carrier for each tag, we select the
    strongest carrier generator among all available candidates.

    Args:
        topology: A TopologyGraph instance to compute the schedule of
        cg_threshold: Minimum acceptable carrier signal strength (default: 30)

    Returns:
        A Schedule instance with the sequential schedule or None is the problem
        is unsatisfiable

    """
    # Create an empty schedule
    schedule = Schedule(topology.active_nodes, len(topology.tags))
        
    # Populate it for all tags in sequence
    for i, (t, h) in enumerate(topology.assignments.items()):
        # Find the best suitable carrier generator
        cg = None
        best_weight = -1
        for n in topology.active_neighbors(t):
            if n == h: 
                continue # The host can't be the carrier generator
            if topology.edges[t, n]['weight'] > best_weight:
                cg = n
                best_weight = topology.edges[t, n]['weight']
        if best_weight <= cg_threshold:
            return None # Unsatisfiable!
        schedule[cg][i] = 'CG'
        schedule[h][i] = t
    return schedule

def run_model(model, data, carrier_thr=0, 
        timeout=300000, command='minizinc'):
    """Execute a MiniZinc model and return the result
    
    Executes the given MiniZinc model on the given data and
    returns the result.
    
    The data is saved to a temporary file and passed to MiniZinc
    for evaluation.
    
    Args:
        model: Filename of the MiniZinc model file
        data: A string containing the contents of the data file
        command: The MiniZinc executable to run. This can be used to 
            run different solvers. i.e: mzn-chuffed, mzn-gecode, etc.

    Returns:
        A string with the output of MiniZinc
    """
    with tempfile.NamedTemporaryFile('w', suffix='.dzn') as data_file:
        data_file.write(data)
        data_file.flush()
        devnull = open(os.devnull, 'w')
        # print(" ".join([command, 
                                 # '--solver', 'Chuffed',
                                 # '--time-limit', f'{timeout}',
                                 # '--cmdline-data', f'CG_THRESHOLD={carrier_thr}',
                                 # model, data_file.name]))
        result = subprocess.run([command, 
                                 '--solver', 'Chuffed',
                                 '--time-limit', f'{timeout}',
                                 '--cmdline-data', f'CG_THRESHOLD={carrier_thr}',
                                 model, data_file.name], 
                                stdout=subprocess.PIPE,
                                stderr=devnull
                               )
        # print(result)
        result_str = result.stdout.decode('utf-8')
        if "Time limit exceeded" in result_str or "UNKNOWN" in result_str:
            raise subprocess.TimeoutExpired(command, timeout/1000)
    return result.stdout.decode('utf-8')

def run_satisfy_model(model, data, carrier_thr=0, opt_value=0,
        timeout=300000, command='minizinc'):
    """Execute a MiniZinc model and return all schedules with given objecive value
    
    Executes the given MiniZinc model on the given data and
    returns all schedules with a given objective function value is any.
    
    The data is saved to a temporary file and passed to MiniZinc
    for evaluation.
    
    Args:
        model: Filename of the MiniZinc model file
        data: A string containing the contents of the data file
        command: The MiniZinc executable to run. This can be used to 
            run different solvers. i.e: mzn-chuffed, mzn-gecode, etc.

    Returns:
        A string with the output of MiniZinc
    """
    with tempfile.NamedTemporaryFile('w', suffix='.dzn') as data_file:
        data_file.write(data)
        data_file.flush()
        devnull = open(os.devnull, 'w')
        result = subprocess.run([command, 
                                 '--solver', 'Chuffed',
                                 '--time-limit', f'{timeout}',
                                 '--cmdline-data', f'CG_THRESHOLD={carrier_thr}',
                                 '--cmdline-data', f'OPTIMAL_OBJ={opt_value}',
                                 '-n 0',
                                 model, data_file.name], 
                                stdout=subprocess.PIPE,
                                stderr=devnull
                               )
        # print(result)
        result_str = result.stdout.decode('utf-8')
        if "Time limit exceeded" in result_str or "UNKNOWN" in result_str:
            raise subprocess.TimeoutExpired(command, timeout/1000)
    return result.stdout.decode('utf-8')

def compute_schedule_mnz(topology, cg_threshold=30, timeout=30000,
        optimize=True,
        model='../discrete_model_modular/model_optimize.mzn',
        **args):
    """Compute the optimal schedule of a topology with MiniZinc

    This is the optimal schedule according to a given MiniZinc model. The
    drawback is that this computation can be very slow for large instances.

    We can optionally apply a heuristic optimization where we remove all
    'irrelevant' active nodes (those not connected to any tag) in the hope of
    lowering the computation time. 

    Args:
        topology: A TopologyGraph instance to compute the schedule of
        cg_threshold: Minimum acceptable carrier signal strength (default: 30)
        timeout: Maximum time to wait for a solution in milliseconds
        optimize: Whether to apply our heuristic optimization (default: True)
        model: Path to the MiniZinc model.

    Returns:
        A Schedule instance with the resulting schedule or None if the problem
        is unsatisfiable

    """
    ret = Schedule(topology.active_nodes)
    if optimize:
        d_graph, non_hosts = topology.clear_irrelevant_nodes()
    else:
        d_graph, non_hosts = topology, []

    for component in d_graph.connected_components():
        problem = component.get_problem_string()
        s = run_model(model, problem, carrier_thr=cg_threshold,
                timeout=timeout) 
        # print(s)
        if 'UNSATISFIABLE' in s:
            return None
        ret.extend(Schedule.parse(s).compressed())
    return ret

def compute_all_schedules_mnz(topology, opt_value=None, cg_threshold=30, timeout=30000,
        optimize=True,
        model_dir='../discrete_model_modular/'):
    if opt_value is None:
        optimal_schedule = compute_schedule_mnz(topology, cg_threshold=cg_threshold, 
                timeout=timeout, optimize=optimize,
                model='/'.join((model_dir, 'model_optimize.mzn')))
        opt_value = optimal_schedule.get_mnz_objectives()['total']
    model = '/'.join((model_dir, 'model_satisfy.mzn'))
    # if optimize:
        # d_graph, non_hosts = topology.clear_irrelevant_nodes()
    # else:
    d_graph, non_hosts = topology, []
    for component in d_graph.connected_components():
        problem = component.get_problem_string()
        s = run_satisfy_model(model, problem, carrier_thr=cg_threshold,
                opt_value=opt_value, timeout=timeout) 
        # print('all_sat', s)
        if 'UNSATISFIABLE' in s:
            return None
    ret = []
    for S in s.split('----------')[:-1]:
        ret.append(Schedule.parse(S).compressed())
    return ret


def compute_schedule_coloring(topology, cg_threshold=30,  **args):
    """Compute a schedule with TagAlong's heuristic algorithm

    This is the scheduling algorithm we developed. It is based on computing a
    graph coloring in the topology's 'conflict' graph. This scheduler is
    sub-optimal but has the advantage of computing in polynomial time.

    Args:
        topology: A TopologyGraph instance to compute the schedule of
        cg_threshold: Minimum acceptable carrier signal strength (default: 30)

    Returns:
        A Schedule instance with the resulting schedule or None if the problem
        is unsatisfiable

    """
    schedule = Schedule(topology.active_nodes)

    g = topology.copy()
    nodes_with_tag = topology.hosts.copy()
    while len(nodes_with_tag) > 0:
        new_sched = Schedule(topology.active_nodes, 1)
        g, _ = g.clear_irrelevant_nodes()
        h = g.conflict_graph()
        d = nx.coloring.greedy_color(h)

        color_members = {c: set() for c in d.values()}
        for k, v in d.items():
            color_members[v].add(k)
            
        scheduled_tags = list()
        interfered_hosts = list()
        scheduled_interrogators = list()
        scheduled = True
        while scheduled:
            scheduled = False
            max_color = None
            max_tags = 0
            for c, nodes in color_members.items():
                n_tags = sum([1 for i in nodes for x in g.neighbors(i) 
                                  if x in g.hosts and 
                                      x not in interfered_hosts and
                                      new_sched[x][0] == 'OFF' and 
                                      g.edges[i, x]['weight'] > cg_threshold])
                if n_tags > max_tags:
                    max_tags = n_tags
                    max_color = c
            if max_color == None:
                if len(scheduled_tags) == 0:
                    # We were unable to schedule any new tags, then the problem
                    # is UNSATISFIABLE
                    return None
                else:
                    break
            
            for cg in color_members[max_color]:
                if cg in scheduled_interrogators:
                    # Potential CG is already scheduled as interrogator, 
                    # skip it 
                    continue
                if set(g.neighbors(cg)) & set(scheduled_interrogators) != set():
                    # Potential CG is neighbor with a scheduled interrogator,
                    # skip it to avoid interfering said neighbor
                    continue
                for n in [x for x in g.neighbors(cg)
                              if x in g.hosts and 
                                x not in interfered_hosts and
                                new_sched[x][0] == 'OFF' and 
                                g.edges[cg, x]['weight'] > cg_threshold]:
                    tag = g.get_hosted_tags(n)[0]
                    new_sched[cg][0] = 'CG'
                    new_sched[n][0] = tag
                    scheduled = True
                    
                    scheduled_interrogators.append(n)
                    scheduled_tags.append(tag)
                    interfered_hosts.extend([g.nodes[x]['host'] for x in g.neighbors(cg) \
                            if x in g.tags])
            del(color_members[max_color])
                
        g.remove_tags(scheduled_tags)
        
        nodes_with_tag = g.hosts
        schedule.append(new_sched)
    return schedule

def parse_ortools_solution(topology, schedule, tag_slots, solver):
    __ACTIONS = ('OFF', 'TX', 'CG')
    n_nodes = topology.graph['n_nodes']
    n_tags = topology.graph['n_tags']
    S = Schedule(topology.active_nodes, n_tags)
    for n in range(n_nodes):
        node_label = topology.node_label_format(n)
        for s in range(n_tags):
            for i in range(3):
                if solver.Value(schedule[(n,s,i)]):
                    S[node_label][s] = __ACTIONS[i]
                    if __ACTIONS[i] == 'TX':
                        for t in topology.get_hosted_tags(node_label):
                            tag_id = topology.tag_id_from_label(t)
                            if solver.Value(tag_slots[(tag_id, s)]):
                                S[node_label][s] = t
                                break
    S = S.compressed()
#     S.__ortools_obj = solver.Value(obj)
    return S

class ScheduleSolutionCollector(cp_model.CpSolverSolutionCallback):
    def __init__(self, schedule, tag_slots, topology):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__schedule = schedule
        self.__tag_slots = tag_slots
        self.__topology = topology
        self.__solution_count = 0
        self.solutions = list()
        
    def on_solution_callback(self):
        self.__solution_count += 1

        S = parse_ortools_solution(self.__topology,
                                   self.__schedule,
                                   self.__tag_slots,
                                   self)
        self.solutions.append(S)

    def solution_count(self):
        return self.__solution_count
    
def prepare_ortools_problem(topology, 
                            cg_threshold=0, 
                            optimize=True,
                            greedy=False):
    n_nodes = topology.graph['n_nodes']
    n_tags = topology.graph['n_tags']
    
    model = cp_model.CpModel()
    
    schedule = {}
    for n in range(n_nodes):
        for s in range(n_tags):
            for i in range(3):
                schedule[(n, s, i)] = model.NewBoolVar(f'sched_{n}_{s}_{i}')
    
    tag_slots = {}
    for t in range(n_tags):
        for s in range(n_tags):
            tag_slots[(t, s)] = model.NewBoolVar(f'ts_{t}_{s}')

    for n in range(n_nodes):
        node_label = topology.node_label_format(n)
        nodes_tag_ids = [topology.tag_id_from_label(t) 
                for t in topology.get_hosted_tags(node_label)]
        for s in range(n_tags):
            model.Add(sum(tag_slots[(t, s)] 
                for t in nodes_tag_ids) == schedule[(n,s,1)])
    
    # Nodes get exactly one role assigned in every timeslot
    for n in range(n_nodes):
        for s in range(n_tags):
            model.Add(sum(schedule[(n, s, i)] for i in range(3)) == 1)

    # Tags get interrogated exactly once
    for t in range(n_tags):
        model.Add(sum(tag_slots[(t, s)] for s in range(n_tags)) == 1)

    # Interrogate co-hosted tags in order
    for n in range(n_nodes):
        node_label = topology.node_label_format(n)
        nodes_tag_ids = sorted([topology.node_id_from_label(x)
                for x in topology.get_hosted_tags(node_label)])
        for (f, s) in zip(nodes_tag_ids[:-1], nodes_tag_ids[1:]):
            for t in range(1, n_tags):
                tmp = model.NewBoolVar(f'enf_{f}_{s}_{t}')
                model.Add(tmp == sum(tag_slots[(f,i)] for i in range(t)))
                model.Add(
                        sum(tag_slots[(s,i)] for i in range(t)) == 0).OnlyEnforceIf(tmp.Not())
        
    # Need exactly one sufficiently strong carrier to interrogate a tag
    for n in range(n_nodes):
        node_label = topology.node_label_format(n)
        nodes_neighbors = topology.active_neighbors(node_label)
        for s in range(n_tags):
            model.Add(
                sum(schedule[(topology.node_id_from_label(o),s,2)] 
                    for o in nodes_neighbors) == 1).OnlyEnforceIf(schedule[(n,s,1)])        
            model.Add(
                sum(schedule[(topology.node_id_from_label(o),s,2)]*topology[node_label][o]['weight']
                    for o in nodes_neighbors) >= cg_threshold).OnlyEnforceIf(schedule[(n,s,1)])
            
    obj_var = model.NewIntVar(1, n_tags, 'num_cg')
    model.Add(obj_var == sum(schedule[(n,s,2)] 
                            for n in range(n_nodes)
                           for s in range(n_tags)))
    

    O2_BASE = 1 if greedy else n_tags+1
    max_obj_2 = sum((i+1)*O2_BASE**(n_tags-1-i) for i in range(n_tags))
    obj_2 = model.NewIntVar(
        0,
        max_obj_2,
        'obj2'
    )
    model.Add(obj_2 == sum(tag_slots[(t, s)]*(s+1)*O2_BASE**(n_tags-1-t)
                           for t in range(n_tags)
                           for s in range(n_tags)))

    max_obj_4 = (n_nodes-1)*(n_tags)
    obj_4 = model.NewIntVar(0, max_obj_4, 'sum_cg_ids')
    model.Add(obj_4 == sum(n*schedule[(n,s,2)] for n in range(n_nodes) for s in range(n_tags)))

    obj_total = model.NewIntVar(1,
                                max_obj_4 + (max_obj_4+1) * max_obj_2 + n_tags*(max_obj_2+1)*(max_obj_4+1),
                               "total_objective")
    model.Add(obj_total == obj_4 + (max_obj_4+1) * obj_2 + obj_var*(max_obj_2+1)*(max_obj_4+1))
    return model, schedule, tag_slots, obj_total


def compute_schedule_ortools(topology, 
                             cg_threshold=0, 
                             timeout=30000,
                             optimize=True, 
                             workers=1, 
                             greedy=False,
                             **args):
    ret = Schedule(topology.active_nodes)
    if optimize:
        d_graph, non_hosts = topology.clear_irrelevant_nodes()
    else:
        d_graph, non_hosts = topology, []

    for component in d_graph.connected_components():
        model, schedule, tag_slots, obj = prepare_ortools_problem(topology, 
                                                                  cg_threshold, 
                                                                  optimize,
                                                                  greedy)
        model.Minimize(obj)
        solver = cp_model.CpSolver()
        if workers <= 0:
            workers += _available_cpu_count()
        solver.parameters.num_search_workers = workers
        solver.parameters.max_time_in_seconds = timeout/1000.
        status = solver.Solve(model)
        # print(solver.StatusName())
        if status == cp_model.INFEASIBLE:
            return None
        if status == cp_model.MODEL_INVALID:
            raise Exception("MODEL_INVALID. Perhaps reduce input size?")
        if status != cp_model.OPTIMAL:
            raise subprocess.TimeoutExpired(f"ortools halted with {solver.StatusName()}", 
                    timeout/1000.)
    #     print(solver.Value(obj))
        S = parse_ortools_solution(topology,
                                      schedule,
                                      tag_slots,
                                      solver)
        ret.extend(S)
        ret.__ortools_obj = solver.Value(obj)
    return ret


def new_prepare_ortools_problem(topology, 
                            cg_threshold=0, 
                            optimize=True,
                            greedy=False):
    n_nodes = topology.graph['n_nodes']
    n_tags = topology.graph['n_tags']
    
    model = cp_model.CpModel()
    
    schedule = {}
    for n in range(n_nodes):
        for s in range(n_tags):
            for i in range(3):
                schedule[(n, s, i)] = model.NewBoolVar(f'sched_{n}_{s}_{i}')
    
    tag_slots = {}
    for t in range(n_tags):
        for s in range(n_tags):
            tag_slots[(t, s)] = model.NewBoolVar(f'ts_{t}_{s}')

    for n in range(n_nodes):
        node_label = topology.node_label_format(n)
        nodes_tag_ids = [topology.tag_id_from_label(t) 
                for t in topology.get_hosted_tags(node_label)]
        for s in range(n_tags):
            model.Add(sum(tag_slots[(t, s)] 
                for t in nodes_tag_ids) == schedule[(n,s,1)])
    
    # Nodes get exactly one role assigned in every timeslot
    for n in range(n_nodes):
        for s in range(n_tags):
            model.Add(sum(schedule[(n, s, i)] for i in range(3)) == 1)

    # Tags get interrogated exactly once
    for t in range(n_tags):
        model.Add(sum(tag_slots[(t, s)] for s in range(n_tags)) == 1)

    # Interrogate co-hosted tags in order
    for n in range(n_nodes):
        node_label = topology.node_label_format(n)
        nodes_tag_ids = sorted([topology.node_id_from_label(x)
                for x in topology.get_hosted_tags(node_label)])
        for (f, s) in zip(nodes_tag_ids[:-1], nodes_tag_ids[1:]):
            for t in range(1, n_tags):
                tmp = model.NewBoolVar(f'enf_{f}_{s}_{t}')
                model.Add(tmp == sum(tag_slots[(f,i)] for i in range(t)))
                model.Add(
                        sum(tag_slots[(s,i)] for i in range(t)) == 0).OnlyEnforceIf(tmp.Not())
        
    # Need exactly one sufficiently strong carrier to interrogate a tag
    for n in range(n_nodes):
        node_label = topology.node_label_format(n)
        nodes_neighbors = topology.active_neighbors(node_label)
        for s in range(n_tags):
            model.Add(
                sum(schedule[(topology.node_id_from_label(o),s,2)] 
                    for o in nodes_neighbors) == 1).OnlyEnforceIf(schedule[(n,s,1)])        
            model.Add(
                sum(schedule[(topology.node_id_from_label(o),s,2)]*topology[node_label][o]['weight']
                    for o in nodes_neighbors) >= cg_threshold).OnlyEnforceIf(schedule[(n,s,1)])
            
    obj_var = model.NewIntVar(1, n_tags, 'num_cg')
    model.Add(obj_var == sum(schedule[(n,s,2)] 
                            for n in range(n_nodes)
                           for s in range(n_tags)))
    

    O2_BASE = 1 if greedy else n_tags+1
    max_obj_2 = sum((n_tags)*O2_BASE**(n_tags-1-i) for i in range(n_tags))
    obj_2 = model.NewIntVar(
        0,
        max_obj_2,
        'obj2'
    )
    model.Add(obj_2 == sum(tag_slots[(t, s)]*(s+1)*O2_BASE**(n_tags-1-t)
                           for t in range(n_tags)
                           for s in range(n_tags)))

    max_obj_4 = (n_nodes-1)*(n_tags)
    obj_4 = model.NewIntVar(0, max_obj_4, 'sum_cg_ids')
    model.Add(obj_4 == sum(n*schedule[(n,s,2)] for n in range(n_nodes) for s in range(n_tags)))

    obj_total = model.NewIntVar(1,
                                max_obj_4 + (max_obj_4+1) * max_obj_2 + n_tags*(max_obj_2+1)*(max_obj_4+1),
                               "total_objective")
    model.Add(obj_total == obj_4 + (max_obj_4+1) * obj_2 + obj_var*(max_obj_2+1)*(max_obj_4+1))
    return model, schedule, tag_slots, obj_total


def new_compute_schedule_ortools(topology, 
                             cg_threshold=0, 
                             timeout=30000,
                             optimize=True, 
                             workers=1, 
                             greedy=False,
                             **args):
    ret = Schedule(topology.active_nodes)
    if optimize:
        d_graph, non_hosts = topology.clear_irrelevant_nodes()
    else:
        d_graph, non_hosts = topology, []

    for component in d_graph.connected_components():
        model, schedule, tag_slots, obj = new_prepare_ortools_problem(topology, 
                                                                  cg_threshold, 
                                                                  optimize,
                                                                  greedy)
        model.Minimize(obj)
        solver = cp_model.CpSolver()
        if workers <= 0:
            workers += _available_cpu_count()
        solver.parameters.num_search_workers = workers
        solver.parameters.max_time_in_seconds = timeout/1000.
        status = solver.Solve(model)
        # print(solver.StatusName())
        if status == cp_model.INFEASIBLE:
            return None
        if status == cp_model.MODEL_INVALID:
            raise Exception("MODEL_INVALID. Perhaps reduce input size?")
        if status != cp_model.OPTIMAL:
            raise subprocess.TimeoutExpired(f"ortools halted with {solver.StatusName()}", 
                    timeout/1000.)
    #     print(solver.Value(obj))
        S = parse_ortools_solution(topology,
                                      schedule,
                                      tag_slots,
                                      solver)
        ret.extend(S)
        ret.__ortools_obj = solver.Value(obj)
    return ret

def compute_all_schedules_ortools(topology, 
                                  opt_value=None, 
                                  cg_threshold=0, 
                                  timeout=30000,
                                  optimize=True,
                                  workers=1,
                                  **args):
    if opt_value is None:
        s = compute_schedule_ortools(topology, 
                                     cg_threshold, 
                                     timeout, 
                                     optimize, 
                                     workers,
                                     greedy=args.get('greedy', False))
        if s is None:
            return None
        opt_value = s.__ortools_obj
    model, schedule, tag_slots, obj = prepare_ortools_problem(topology, 
                                                          cg_threshold, 
                                                          optimize, greedy=args.get('greedy', False))
    model.Add(obj == opt_value)
    
    solver = cp_model.CpSolver()
#     solver.parameters.num_search_workers = workers
    solver.parameters.max_time_in_seconds = timeout/1000.
    solution_collector = ScheduleSolutionCollector(schedule, tag_slots, topology)
    status = solver.SearchForAllSolutions(model, solution_collector)
    return solution_collector.solutions

def new_compute_all_schedules_ortools(topology, 
                                  opt_value=None, 
                                  cg_threshold=0, 
                                  timeout=30000,
                                  optimize=True,
                                  workers=1,
                                  **args):
    if opt_value is None:
        s = new_compute_schedule_ortools(topology, 
                                     cg_threshold, 
                                     timeout, 
                                     optimize, 
                                     workers)
        if s is None:
            return None
        opt_value = s.__ortools_obj
    model, schedule, tag_slots, obj = new_prepare_ortools_problem(topology, 
                                                          cg_threshold, 
                                                          optimize)
    model.Add(obj == opt_value)
    
    solver = cp_model.CpSolver()
#     solver.parameters.num_search_workers = workers
    solver.parameters.max_time_in_seconds = timeout/1000.
    solution_collector = ScheduleSolutionCollector(schedule, tag_slots, topology)
    status = solver.SearchForAllSolutions(model, solution_collector)
    return solution_collector.solutions


#TODO Maybe discover these automatically?
SCHEDULER_FUNCTIONS = {
        'sequential': compute_schedule_sequential,
        'minizinc'  : compute_schedule_mnz,
        'tagalong'  : compute_schedule_coloring,
        'ortools'   : compute_schedule_ortools
        }
SCHEDULERS = tuple(SCHEDULER_FUNCTIONS.keys())
