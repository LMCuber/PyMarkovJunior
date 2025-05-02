from .settings import *
from .wfc import WfcNode

import pygame
from pygame.time import get_ticks as ticks
import sys
import cProfile
import xml.etree.ElementTree as ET
import json
import numpy as np
import random
import time
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import convolve2d
from fractions import Fraction
from pathlib import Path
from itertools import chain
from math import sqrt


# colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_GRAY = (40, 40, 40)
LIGHT_GRAY = (200, 200, 200)

# constants
pygame.init()
WIDTH = 2 ** 9
HEIGHT = 2 ** 9
pygame.display.set_caption("Markov Junior")
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
TARG_FPS = 60
font = pygame.font.SysFont("Courier New", 30)


# FUNCTIONS
def flatten2d(iterable):
    return sum(iterable, [])


def dict_to_pairs(data):
    if isinstance(data, dict):
        return [[key, dict_to_pairs(value)] for key, value in data.items()]
    elif isinstance(data, list):
        return [dict_to_pairs(item) for item in data]
    else:
        return data


def uniques(lst):
    result = []
    for item in lst:
        if not any(np.array_equal(item, x) for x in result):
            result.append(item)
    return result


# CLASSES
class Markov:
    def __init__(self, rules_path, grid_size):
        # very excellent variable naming
        self.grid_size = grid_size
        self.width, self.height = self.grid_size
        self.size = WIDTH / self.width
        self.running = True
        # initialization of important stuff
        self.auto_update = True
        self.init_hashmaps()
        self.parse_rules(rules_path)
        self.image = pygame.Surface((self.width, self.height))
        self.updated_indices = [
            [y, 1, x, 1, np.array([[self.grid[y, x]]])] 
            for y in range(self.grid.shape[0]) 
            for x in range(self.grid.shape[1])
        ]
        #
        self._window_cache = {}
        #
        self.last_update = ticks()
    
    def render_to_surf(self, surf):
        for y_start, y_end, x_start, x_end, out_pattern in self.updated_indices:
            for dy in range(out_pattern.shape[0]):
                for dx in range(out_pattern.shape[1]):
                    tile = out_pattern[dy, dx]
                    if tile != "*":
                        color = self.color_map[tile]
                        self.image.set_at((x_start + dx, y_start + dy), color)
        surf.blit(self.image, (0, 0))
        self.updated_indices = []
    
    def update(self):
        if ticks() - self.last_update >= 0:
            self.process_rules()
            self.last_update = ticks()
        
    def palettize(self, src_color):
        closest_dist = float("inf")
        closest_color = None
        for color_code, rgb in self.color_map.items():
            dist = sqrt((src_color[0] - rgb[0]) ** 2 + (src_color[1] - rgb[1]) ** 2 + (src_color[2] - rgb[2]) ** 2)
            if dist < closest_dist:
                closest_dist = dist
                closest_color = color_code
        return closest_color
        
    def init_hashmaps(self):
        self.color_map = {
            "B": pygame.Color("#000000"),
            "I": pygame.Color("#1D2B53"),
            "P": pygame.Color("#7E2553"),
            "E": pygame.Color("#008751"),
            "N": pygame.Color("#AB5236"),
            "D": pygame.Color("#5F574F"),
            "A": pygame.Color("#C2C3C7"),
            "W": pygame.Color("#FFF1E8"),
            "R": pygame.Color("#FF004D"),
            "O": pygame.Color("#FFA300"),
            "Y": pygame.Color("#FFEC27"),
            "G": pygame.Color("#00E436"),
            "U": pygame.Color("#29ADFF"),
            "S": pygame.Color("#83769C"),
            "K": pygame.Color("#FF77A8"),
            "F": pygame.Color("#FFCCAA"),
        }
        self.symmetries = {
            "()":     (True, False, False, False, False, False, False, False),
            "(x)":    (True, True,  False, False, False, False, False, False),
            "(y)":    (True, False, False, False, False, True,  False, False),
            "(x)(y)": (True, True,  False, False, True,  True,  False, False),
            "(xy+)":  (True, False, True,  False, True,  False, True,  False),
            "(xy)":   (True, True,  True,  True,  True,  True,  True,  True),
        }
        self.unions = {}
    
    def get_neighborhood_kernel(self, kernel_size, type_):
        n = 2 * kernel_size + 1
        if type_ == "Moore":
            return [
                [0 if (x == kernel_size and y == kernel_size) else 1 for x in range(n)]
                for y in range(n)
            ]
        elif type_ == "VonNeumann":
            return [
                [1 if ((x == kernel_size) ^ (y == kernel_size)) else 0 for x in range(n)]
                for y in range(n)
            ]
    
    def apply(self, indices: list[list[int, int, int, int, str]]):
        for index in indices:
            # update updated_indices list for rendering purposes
            self.updated_indices.append(index)
            # update the map if forall non wildcard
            for yo in range(index[1] - index[0]):
                for xo in range(index[3] - index[2]):
                    color = index[4][yo][xo]
                    if color != "*":
                        self.grid[index[0] + yo, index[2] + xo] = color
        
    def create_conflict(self, y, x):
        self.conflicts.add((y, x))
        
    def reset_conflicts(self):
        self.conflicts = set()
    
    def parse_range(self, value):
        if ".." in value:
            # value is a range
            return list(range(int(value.split("..")[0]), int(value.split("..")[1]) + 1))
        else:
            # value is a single number
            return [int(value)]
    
    def match(self, rule) -> bool:
        # set the relevant rule information to handy variables
        tag = rule["tag"]
        in_ = rule["in"]
        out = rule["out"]
        sym = rule.get("symmetry", "(xy)")

        # apply unions before generating 2D pattern
        # TODO: unions

        # get the original in and out patterns in 2D space
        in_pattern = np.array([list(row) for row in in_.split("/")])
        out_pattern = np.array([list(row) for row in out.split("/")])

        # apply symmetries
        augmented_in_patterns = [in_pattern]  # augmented means a symmetrical transform
        augmented_out_patterns = [out_pattern]
        if in_pattern.shape != (1, 1):
            # if shape is NOT a 1x1 pixel, then continue
            symmetry_indices = self.symmetries[sym]
            if symmetry_indices[1]:
                augmented_in_patterns.append(np.fliplr(in_pattern))
                augmented_out_patterns.append(np.fliplr(out_pattern))
            if symmetry_indices[2]:
                augmented_in_patterns.append(np.rot90(in_pattern))
                augmented_out_patterns.append(np.rot90(out_pattern))
            if symmetry_indices[3]:
                augmented_in_patterns.append(np.fliplr(np.rot90(in_pattern)))
                augmented_out_patterns.append(np.fliplr(np.rot90(out_pattern)))
            if symmetry_indices[4]:
                augmented_in_patterns.append(np.rot90(in_pattern, 2))
                augmented_out_patterns.append(np.rot90(out_pattern, 2))
            if symmetry_indices[5]:
                augmented_in_patterns.append(np.fliplr(np.rot90(in_pattern, 2)))
                augmented_out_patterns.append(np.fliplr(np.rot90(out_pattern, 2)))
            if symmetry_indices[6]:
                augmented_in_patterns.append(np.rot90(in_pattern, 3))
                augmented_out_patterns.append(np.rot90(out_pattern, 3))
            if symmetry_indices[7]:
                augmented_in_patterns.append(np.fliplr(np.rot90(in_pattern, 3)))
                augmented_out_patterns.append(np.fliplr(np.rot90(out_pattern, 3)))

        # make sure duplicates are removed (BUG: it mismatches the lengths of input and output so don't use yet)
        # augmented_in_patterns = uniques(augmented_in_patterns)
        # augmented_out_patterns = uniques(augmented_out_patterns)

        # randomly shuffle these symmetries
        shuffle_indices = list(range(len(augmented_in_patterns)))
        random.shuffle(shuffle_indices)

        augmented_in_patterns = [augmented_in_patterns[i] for i in shuffle_indices]
        augmented_out_patterns = [augmented_out_patterns[i] for i in shuffle_indices]

        lh, lw = self.grid.shape
        indices = []

        # cache enabled sliding view getter
        def get_sliding_window(key):
            if key in self._window_cache:
                return self._window_cache[key]
            else:
                self._window_cache[key] = sliding_window_view(self.grid, (sh, sw))
                return self._window_cache[key]

        # check what type of pattern matching we have to do
        if tag == "convolution":
            # neighborhood checks
            num_permitted_neighbors = self.parse_range(rule["sum"])
            values = rule["values"]

            mask = np.isin(self.grid, list(values)).astype(int)

            kernel = self.get_neighborhood_kernel(rule["kernel"], rule["neighborhood"])
            num_neighbors = convolve2d(mask, kernel, mode="same", boundary="fill", fillvalue=0)

            # Step 3: Find where the original grid is 'A' and has >=6 D neighbors
            # Note: Subtract 1 if you want to exclude the center cell itself (but here we only care about neighbors of A)
            condition = (self.grid == in_) & np.isin(num_neighbors, num_permitted_neighbors)

            # Step 4: Get the positions
            matched_positions = np.argwhere(condition)
            # boolean indexing
            for y, x in matched_positions:
                indices.append([y, y + 1, x, x + 1, out_pattern])
        else:
            # iterate over all possible rotated input shapes to check whether they match in the grid
            for i, augmented_in_pattern in enumerate(augmented_in_patterns):
                # the size of the pattern window
                sh, sw = augmented_in_pattern.shape

                # checking whether the pattern matches, including wildcard (*)
                mask = augmented_in_pattern != "*"  # make sure to match asterisk with anything
                windows = get_sliding_window((sh, sw))
                comparison = (windows == augmented_in_pattern) | ~mask
                matches = comparison.all(axis=(-2, -1))
                matched_positions = np.argwhere(matches)

                # iterate through all found matches in the grid
                for y, x in matched_positions:
                    any_conflicts = False
                    augmented_out_pattern = augmented_out_patterns[i].copy()

                    # make sure to replace target asterisks
                    for dy in range(sh):
                        for dx in range(sw):
                            # make sure no conflicts arise when using <all>
                            if tag == "all":
                                # check if target pixel is saved as a conflict
                                if (y + dy, x + dx) in self.conflicts and augmented_out_pattern[dy, dx] != "*":
                                    any_conflicts = True
                                # if output pixel is left alone, it doesn't cause conflicts
                                if augmented_out_pattern[dy, dx] != "*":
                                    self.create_conflict(y + dy, x + dx)

                    if not any_conflicts:
                        indices.append([y, y + sh, x, x + sw, augmented_out_pattern])
                
        return indices

    def process_rules(self):
        self.root_node.process()

    def parse_rules(self, rules_path):
        #
        self.root_node = None
        #
        tree = ET.parse(rules_path)
        # parse metadata
        root = tree.getroot()

        def parse(node, parent):
            attrib = node.attrib | {"tag": node.tag}
            node_obj = None
            if node.tag == "sequence":
                node_obj = SequenceNode(parent, attrib, self)
            elif node.tag == "markov":
                node_obj = MarkovNode(parent, attrib, self) 
            elif node.tag == "one":
                node_obj = OneNode(parent, attrib, self)
            elif node.tag == "all":
                node_obj = AllNode(parent, attrib, self)
            elif node.tag == "prl":
                node_obj = PrlNode(parent, attrib, self)
            elif node.tag == "rule":
                node_obj = RuleNode(parent, attrib, self)
            elif node.tag == "convolution":
                node_obj = ConvolutionNode(parent, attrib, self)
            elif node.tag == "wfc":
                node_obj = WfcNode(parent, attrib, self)

            elif node.tag == "union":
                self.unions[attrib["symbol"]] = attrib["values"]

            if node_obj is not None:
                if self.root_node is None:
                    self.root_node = node_obj

                if parent is not None:
                    parent.add_children(node_obj)
                for child in node:
                    parse(child, node_obj)

        parse(root, None)


class Node:
    def __init__(self, parent, data, markov):
        self.parent = parent
        # setup data correctly
        self.data = data
        self.data["steps"] = int(self.data["steps"]) if "steps" in self.data else float("inf")
        self.data["p"] = Fraction(self.data["p"]) if "p" in self.data else 1
        if "kernel" in self.data:
            self.data["kernel"] = int(self.data["kernel"])
        #
        self.children = []
        self.markov = markov
        self.num_matches = 0
        values = self.data.get("values", [])
        # init grid if a correct "values" is passed
        if values and self.data["tag"] and (self.parent is None or self.parent.data["tag"] != "convolution"):
            self.markov.grid = np.full((self.markov.height, self.markov.width), values[0])
            self.markov.reset_conflicts()
            if is_true(self.data.get("origin", "")):
                self.markov.grid[self.markov.height // 2, self.markov.width // 2] = values[1]

        # inherited attributes
        if self.parent is not None:
            for attr in ("symmetry", "kernel"):
                if not attr in self.data:
                    if attr in self.parent.data:
                        self.data[attr] = self.parent.data[attr]
        
    def add_children(self, *children):
        self.children.extend(children)


"""
How the Sequence node parses return values from their cihldren:
-- True: the child found a match; there is more work to do, so don't continue to next node; stay
-- False: child failed a match; the processing is done; can continue to next child node
"""
class SequenceNode(Node):
    def __init__(self, parent, data, markov=None):
        self.child_index = 0
        Node.__init__(self, parent, data, markov)
    
    def reset(self):
        pass
    
    def process(self):
        try:
            self.children[self.child_index]
        except IndexError:
            # all children exhausted
            return False
        else:
            match = self.children[self.child_index].process()

        if not match:
            self.child_index += 1

        return True


"""
How the Markov node parses return values from their cihldren:
-- True: the child found a match; the rest of children don't need to be checked; reset to top child
-- False: child failed a match; continuet to the next child. If none matched, finish matching
"""
class MarkovNode(Node):
    def process(self):
        for i, child in enumerate(self.children):
            if isinstance(child, SequenceNode):
                child.reset()
            matched = child.process()
            if matched:
                # first match is found; repeat process
                return True

        # not a single match, halt
        return False


class ConvolutionNode(Node):
    def __init__(self, *args, **kwargs):
        Node.__init__(self, *args, **kwargs)
        if "kernel" not in self.data:
            self.data["kernel"] = 1
        if "neighborhood" not in self.data:
            raise ValueError('Missing required attribute "neighborhood" for the <convolution> tag')
    
    def process(self):
        matched_any = False
        matched_indices = []
        for child in self.children:
            for match in child.get_matches():
                matched_indices.append(match)
                matched_any = True
        
        self.markov.apply(matched_indices)

        if matched_indices:
            return True
        else:
            return False

"""
Only call RuleNode's __init__ function when it is a RuleNode and doesn't inherit it. Otherwise, call Node's __init__
"""
class RuleNode(Node):
    def __init__(self, parent, *args, **kwargs):
        Node.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        # because rule nodes are ambiguous and depend on their parents node, for example <one> / <all> / <prl> etc.
        self.data["tag"] = self.parent.data["tag"]
        if "neighborhood" in self.parent.data:
            self.data["neighborhood"] = self.parent.data["neighborhood"]
    
    def get_matches(self):
        return self.markov.match(self.data)


class OneNode(RuleNode):
    def __init__(self, *args, **kwargs):
        Node.__init__(self, *args, **kwargs)

    def process(self):
        matched_data = []
        if self.children:
            # if has children, get their matches
            for child in self.children:
                matched_data.append([child.data["p"], child.get_matches()])
        else:
            # get own match (<one> is standalone)
            matched_data.append([self.data["p"], self.get_matches()])

        if any(data[1] for data in matched_data):
            flattened_matches = flatten2d([data[1] for data in matched_data])
            random_match = random.choice(flattened_matches)
            self.markov.apply([random_match])
            #
            self.num_matches += 1
            if self.num_matches >= self.data["steps"]:
                # maximum number of matches is found
                return False

            # at least 1 match is found, so not exhausted
            return True
        
        # found not a single match -> finish
        return False


class AllNode(RuleNode):
    def __init__(self, parent, *args, **kwargs):
        Node.__init__(self, parent, *args, **kwargs)
        
    def process(self):
        """
        Get all matches from children and save them per child. Then if probabilities for each match happen, apply those matches.
        """

        matched_data: list[list[int, list]] = []
        self.markov.reset_conflicts()
        
        if self.children:
            # if has children, get their matches
            for child in sorted(self.children, key=lambda x: random.random()):
                matched_data.append([child.data["p"], child.get_matches()])
        else:
            # get own match
            matched_data.append([self.data["p"], self.get_matches()])
        
        for prob, matches in matched_data:
            for match in matches:
                if random.random() <= prob:
                    self.markov.apply([match])

        # check whether any actual matches reside in the matched_data list
        if any(data[1] for data in matched_data):
            self.num_matches += 1
            if self.num_matches >= self.data["steps"]:
                # maximum number of matches is found
                return False
            return True
        else:
            return False


class PrlNode(AllNode):
    pass
