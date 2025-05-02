from .settings import *

import pygame
import sys
import cProfile
from pprint import pprint
import random
from functools import partial
from typing import Any
from threading import Thread
import numpy as np


# functions
def _emoji(kernel):
    ret = ""
    for row in kernel:
        for color in row:
            ret += {
                (255, 255, 255): "⬜",
                (0, 0, 0): "⬛",
                (237, 28, 36): "🟥",
            }[color]
        ret += "\n"
    ret = ret.removesuffix("\n")
    return ret


def _rotate_kernel_90(kernel):
    transposed_kernel = list(zip(*kernel))
    rotated_kernel = [list(row)[::-1] for row in transposed_kernel]
    return rotated_kernel


# classes
KernelType = TileType = list[list[Any]]
RulesType = dict[KernelType, list[set[KernelType]]]
MapType = list[list[TileType]]


class WfcNode:
    def __init__(self, parent, data, markov):
        self.parent = parent
        self.markov = markov
        self.data = data
        self.random = random.Random()
        # setup the rules and grid
        self._load_source_array_from_surf()
        sys.setrecursionlimit(int(self.data["recursion-limit"]))
        self.direction_map = {0: 2, 2: 0, 3: 1, 1: 3}
        self.max_depth = int(self.data["max-depth"])
        self.kernel_size = int(self.data["n"])
        self.rotate = is_true(self.data.get("rotate", "False"))
        self.rules: RulesType = {}
        self.grid: MapType = []
        self.kernel_frequencies: dict[KernelType, int] = {}
        # * process the given source image
        # * make up rules hashmap
        # * populate with tiles ready to be collapsed
        self._extract_kernels()
        self._generate_rules()
        self._populate_map()
    
    def process(self):
        # self.collapse()
        Thread(target=self.collapse, daemon=True).start()
        return False
    
    def collapse(self):
        self._populate_map()
        self._collapse(first=True)
        
    def _update_markov(self):
        for y, row in enumerate(self.grid):
            for x, tile in enumerate(row):
                center_pixel = tile.img[1][1]
                # self.markov.apply([[y, 1, x, 1, np.array([[center_pixel]])]])
                return
    
    def _load_source_array_from_surf(self):
        source_img = pygame.image.load(self.data["sample"])
        self.source_array = []
        for y in range(source_img.height):
            row = []
            for x in range(source_img.width):
                color = source_img.get_at((x, y))
                color_code = self.markov.palettize(color)
                row.append(color_code)
            self.source_array.append(row)
    
    def _collapse(self, first=False):
        # make all tiles unchecked
        for row in self.grid:
            for tile in row:
                tile.checked = False
        if first:
            # find initial center tile to collapse
            x = self.random.randrange(self.markov.width)
            y = self.random.randrange(self.markov.height)
            tile = self.grid[y][x]
        else:
            # find the tile with the lowest entropy and collapse it
            sorted_tiles = [tiles for row in self.grid for tiles in row if not tiles.collapsed]
            if not sorted_tiles:
                # all tiles have been collapsed, update the markov grid and return
                return True
            # tile = min(sorted_tiles, key=lambda x: len(x.options))
            tile = self._cmin(sorted_tiles)
        # try to collapse the tile
        success = tile.collapse()
        if not success:
            # there were 0 options left, TODO: backtracking HACK: restart
            self.collapse()
        tile.reduce_neighbors()
        # RECURSION !!!
        self._collapse()
    
    def _cmin(self, tiles):
        min_choice = None
        min_len = float("inf")
        
        for tile in tiles:
            if tile.options_len < min_len:
                min_len = tile.options_len
                min_choice = tile

        return min_choice
    
    def _get_kernel_border(self, kernel, index):
        match index:
            case 0:
                return [kernel[0], kernel[1]]
            case 1:
                return [[kernel[i][1], kernel[i][2]] for i in range(3)]
            case 2:
                return [kernel[1], kernel[2]]
            case 3:
                return [[kernel[i][0], kernel[i][1]] for i in range(3)]

    def _generate_rules(self):
        for kernel in self.all_kernels:
            if kernel in self.rules:
                continue
            self.rules[kernel] = []
            for i in range(4):
                self.rules[kernel].append(set())
            for other in self.all_kernels:
                for i in range(4):
                    other_i = self.direction_map[i]
                    border = self._get_kernel_border(kernel, i)
                    other_border = self._get_kernel_border(other, self.direction_map[i])
                    if border == other_border:
                        self.rules[kernel][i].add(other)
    
    def _extract_kernels(self):
        def extract(arr):
            ret = set()
            height = len(arr)
            width = len(arr[0])
            for y in range(height):
                for x in range(width):
                    kernel = tuple(
                        tuple(arr[(y + dy) % height][(x + dx) % width] for dx in range(-1, 2))
                        for dy in range(-1, 2)
                    )
                    # update the frequency table
                    self.kernel_frequencies[kernel] = self.kernel_frequencies.get(kernel, 0) + 1
                    ret.add(kernel)
            return ret

        self.all_kernels = set()
        self.all_kernels.update(extract(self.source_array))
        if self.rotate:
            self.all_kernels.update(extract(_rotate_kernel_90(self.source_array)))
    
    def _populate_map(self):
        self.grid = [[_Tile(y, x, self) for x in range(self.markov.width)] for y in range(self.markov.height)]


class _Tile:
    def __init__(self, y, x, wfc):
        self.y = y
        self.x = x
        self.wfc = wfc
        self.img = None
        self.options = set(self.wfc.all_kernels)
        self.options_len = len(self.options)
        self.collapsed = False
        self.checked = False
        self.visit = -1
    
    def collapse(self):
        self.collapsed = True
        if self.options:
            self.img = self.wfc.random.choices(list(self.options), weights=[self.wfc.kernel_frequencies[kernel] for kernel in self.options], k=1)[0]
            center_pixel = self.img[1][1]
            # update the markov grid as well as the wfc grid
            self.wfc.markov.apply([[self.y, self.y + 1, self.x, self.x + 1, np.array([[center_pixel]])]])
        else:
            return False

        self.options = [self.img]
        self.options_len = 1

        return True

    def reduce_neighbors(self, depth=0):
        if depth > self.wfc.max_depth:
            return

        if self.checked:
            return
            
        # get the neighbors
        self.checked = True
        # offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        for i, (yo, xo) in enumerate(offsets):
            if self.x + xo < 0 or self.x + xo >= len(self.wfc.grid[0]) or self.y + yo < 0 or self.y + yo >= len(self.wfc.grid):
                continue
            neighboring_tile = self.wfc.grid[self.y + yo][self.x + xo]
            success = neighboring_tile.update_options(self, i)
            if success:
                neighboring_tile.reduce_neighbors(depth + 1)
    
    def update_options(self, prev, direc):
        if self.collapsed:
            return False
        
        # only save hypothetical options that the tile already has, thus the intersection
        hypothetical_options = set()
        for option in prev.options:
            hypothetical_options.update(self.wfc.rules[option][direc])
        self.options = set.intersection(self.options, hypothetical_options)
        self.options_len = len(self.options)

        if not self.options:
            # found a contradiction, so backtracking is needed
            ...

        return True
