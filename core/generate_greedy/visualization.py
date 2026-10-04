import re 

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter


class TopologyVisualizer(object):
    """Draw a network topology using networkx's drawing functionality
    
    Params:
        node_color: The color to use to draw active nodes
        tag_color: Color to draw passive tags
    """
    SHAPES = {'tag': 's', 'active': 'o', 'shaded': 'o'}
    LABELS = {'tag': 'Sensor tag', 'active': 'Regular node', 'link': 'Wireless link'}
    # if style is not None:
        # COLORS = { 'tag': style.COLORS[0], 'active': style.COLORS[2], 'shaded': 'gray'}
        # COLORS = { 'tag': 'lightgray', 'active': 'gray', 'shaded': 'w'}
    # else:
    # COLORS = {'tag': 'cyan', 'active': 'red', 'shaded': 'gray'}
    COLORS = {'tag': '#548235', 'active': '#0070C0', 'shaded': 'gray', 'nodefont': 'white'}

    @staticmethod
    def show(topology, defer=False, legend=False, minimalist=True,
             in_figsize=(6.4, 4.8), in_nodesize=500, in_fontsize=10):
        nodePos = topology.position_nodes()
        
        # The rest of the code here attempts to automate the whole process by
        # first determining how many different node classes (according to
        # attribute 's') exist in the node set and then repeatedly calling
        # draw_networkx_node for each. Perhaps this part can be optimised further.

        # Get all distinct node classes according to the node type attribute
        node_types = list(set((topology.nodes[n]['type'] for n in topology.nodes)))
        
        _, ax = plt.subplots(figsize=in_figsize)
        ax.axis('off')

        artists = list()
        #For each node class...
        for t in node_types:
            #...filter and draw the subset of nodes with the same symbol in the
            # positions that are now known through the layout.  
            tmp = nx.draw_networkx_nodes(
                topology,
                nodePos, ax=ax,
                node_shape = TopologyVisualizer.SHAPES[t], 
                node_color = TopologyVisualizer.COLORS[t],
                nodelist = [n for n in topology.nodes if topology.nodes[n]['type'] == t],
                linewidths=1.8, node_size=in_nodesize
            )
            tmp.set_edgecolor('k')
            artists.append(tmp)
        nx.draw_networkx_labels(topology, nodePos, font_color=TopologyVisualizer.COLORS['nodefont'], ax=ax, font_size=in_fontsize,
                                font_weight='bold')
        
        #Finally, draw the edges between the nodes
        ge = topology.copy()
        if minimalist:
            remove_edges = list()
            for t in topology.tags:
                for n in topology.neighbors(t):
                    if n != topology.nodes[t]['host']:
                        remove_edges.append((t, n))
            ge.remove_edges_from(remove_edges)
        artists.append(nx.draw_networkx_edges(ge, nodePos, ax = ax))
        if legend:
            node_types.append('link')
            plt.legend(artists, [TopologyVisualizer.LABELS[t] for t in node_types], markerscale=0.4,
                    loc='upper left')
        if not defer:
            plt.show()


class ScheduleVisualizer(object):
    LABELS = {'TX': 'Int.', 'RX': 'Reply', 'CG':'Carrier'}
    # N_NODES = len(schedule)
    # if style is not None:
        # COLORS = { 'TX': style.COLORS[0], 'RX': style.COLORS[1], 'CG': style.COLORS[2]}
        # HATCHES = { 'TX': style.HATCHES[1], 'RX': style.HATCHES[2], 'CG': style.HATCHES[3]}
    # else:
    COLORS = {'TX': 'blue', 'RX': 'green', 'CG': 'red'}
    HATCHES = {'TX': '', 'RX': '', 'CG': ''}

    @staticmethod
    def _sorted_node_list(nodes, reverse=False):
        RE_NODE_NUMBERS = re.compile(r'.*\D(\d+)')

        return sorted(nodes, 
                key=lambda x: int(RE_NODE_NUMBERS.match(x).group(1)),
                reverse=reverse)

    @staticmethod
    def show(schedule, defer=False, regular_slots=0, 
            style=None, legend=False, *args): 
        """Draw a graphical representation of the given schedule
        
        Args:
            schedule: A Schedule object
        """
        if schedule is None:
            return
        if not defer:
            fig = plt.figure()
            ax = fig.add_axes([0.1,0.1,0.85*(2/3 if legend else 1),0.85])
        else:
            ax = plt.gca()
        nodes = ScheduleVisualizer._sorted_node_list(
                    schedule.active_nodes, reverse=True) 
        n_nodes = len(nodes)
        n_slots = len(schedule)
        if regular_slots > 0:
            r = patches.Rectangle((n_slots, -0.5), regular_slots, n_nodes,
                    alpha=0.5, facecolor='lightgray') 
            ax.add_patch(r)
            ax.text(n_slots + regular_slots/2, n_nodes-1, "Normal\nSchedule",
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=14, color='k', bbox=dict(boxstyle='square',
                        fc='white'))
        for i, node in enumerate(nodes):
            for j, slot in enumerate(schedule[node]):
                if slot != 'OFF':
                    r = patches.Rectangle(
                            (j, i-0.25), 1, 0.5, 
                            facecolor=ScheduleVisualizer.COLORS['RX'] if slot in schedule.tags else ScheduleVisualizer.COLORS[slot], 
                            hatch=ScheduleVisualizer.HATCHES['RX'] if slot in schedule.tags else ScheduleVisualizer.HATCHES[slot], 
                            label=slot if slot in schedule.tags else ScheduleVisualizer.LABELS[slot], alpha=.75
                    )
                    ax.add_patch(r)
                    ax.text(j+0.5, i, 
                            '{}'.format(slot if slot in schedule.tags else  'CG'), 
                            horizontalalignment='center',
                            verticalalignment='center',
                            color='w')
                    
        ax.tick_params(axis='x', which='minor', bottom=True)
        ax.xaxis.set_minor_locator(FixedLocator(0.5 + np.arange(n_slots+regular_slots+1)))
        ax.xaxis.set_major_formatter(NullFormatter())
        slot_labels = list(('{}'.format(i) for i in range(1, n_slots+regular_slots+1)))
        if regular_slots > 0:
            slot_labels[-1] += '$\\ldots$'
        ax.xaxis.set_minor_formatter(FixedFormatter(slot_labels))
        ax.tick_params(axis='x', which='minor', length=0)

        ax.set_xlim(0, n_slots)
        ax.set_ylim(-0.5, n_nodes-0.5)
        ax.set_yticks(range(n_nodes))
        ax.set_yticklabels(nodes)
        ax.set_xticks(range(0, n_slots+1+regular_slots))
        ax.grid(True)
        ax.set_xlabel('Slot')
        ax.set_ylabel('Node')
        if legend:
            plt.legend(loc=(1.04,0), fontsize=10)
        if not defer:
            plt.show()
