#!/usr/bin/env python3

import time
import numpy as np
import cv2

factor = 10

cost_map_image = cv2.imread("map_data/cost_map.jpg")
cost_map_image = cv2.resize(cost_map_image, (int(1280/factor), int(720/factor)))
width = cost_map_image.shape[1]
height = cost_map_image.shape[0]
lambda_ = 1
image_map = np.zeros((height, width, 1), np.uint8)
image_map = cost_map_image[:,:,0]

class GridSpace():
    def __init__(self):
        self.occupied = False
        self.closed = False
        self.expand_value = float("inf")
        self.heuristic = 0
        self.move_cost = 1

    def close(self):
        self.closed = True

    def expand(self, expansion):
        self.expand_value = expansion

class Node():
    def __init__(self, x = 0, y = 0, cost = float("inf")):
        self.x = x
        self.y = y
        self.cost = cost

    def print(self):
        print("X: %s, Y: %s, Cost: %s " % (self.x, self.y, self.cost))

class Direction():
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = name

def search_list_for_node(list, search_node):
    found = False
    # print("Contents of search list:")
    # for node in list:
    #     node.print()
    # print("Looking for:")
    # search_node.print()
    i=0
    for node in list:
        if node.x == search_node.x and node.y == search_node.y:
            found = True
            break
        else:
            i+=1
    if found:
        return True, i
    else:
        return False, 0

def main(start_x, start_y, goal_x, goal_y, occupied_area):
    start_node = Node(int(start_x/factor), int(start_y/factor), 0)
    neighbour_node = Node()
    goal_node = Node(int(goal_x/factor), int(goal_y/factor))
    gridmap_ = []
    directions = []
    directions.append(Direction(-1, 0, "left"))
    directions.append(Direction(0, 1, "down"))
    directions.append(Direction(1, 0, "right"))
    directions.append(Direction(0, -1, "up"))

    for y in range(0, height):
        grid_x = []
        for x in range (0, width):
            gridspace = GridSpace()
            grid_x.append(gridspace)
        gridmap_.append(grid_x)

    for y in range(0, height):
        for x in range (0, width):
            gridmap_[y][x].heuristic = (abs(y-goal_node.y)+abs(x-goal_node.x))
            gridmap_[y][x].move_cost = image_map[y,x]
            if x == 0 or y == 0 or x == width-1 or y == height-1:# or image_map[y,x] ==255:
                gridmap_[y][x].occupied = True
                image_map[y,x] = 255
    if occupied_area:
        # image_map[int(occupied_area[1]/factor), int(occupied_area[0]/factor)] = 255
        for y in range (int(occupied_area[1]/factor-100/factor), int(occupied_area[1]/factor+100/factor)):
            for x in range (int(occupied_area[0]/factor-100/factor), int(occupied_area[0]/factor+100/factor)):
                gridmap_[y][x].occupied = True

    gridmap_[start_node.y][start_node.x].close()
    gridmap_[start_node.y][start_node.x].expand(0)
    open_list = []
    open_list.append(start_node)

    expansion = 1
    print("Start node: ", end="")
    start_node.print()
    print("Goal node: ", end="")
    goal_node.print()
    print("Searching...")
    while True:
        open_list.sort(key=lambda x: x.cost, reverse=True)
        lowest_cost_node = open_list.pop()
        # print("Searching around:")
        # lowest_cost_node.print()
        image_map[lowest_cost_node.y, lowest_cost_node.x] = 200
        # cv2.imshow("Image", image_map)
        # cv2.waitKey(1)
        gridmap_[lowest_cost_node.y][lowest_cost_node.x].closed = True
        if (lowest_cost_node.x == goal_node.x and lowest_cost_node.y == goal_node.y):
            print("Goal reached!")
            break
        else:
            for d in directions:
                # print("Contents of open_list:")
                # for node in open_list:
                #     node.print()
                neighbour_node.x = lowest_cost_node.x+d.x
                neighbour_node.y = lowest_cost_node.y+d.y
                neighbour_node.cost = lowest_cost_node.cost+gridmap_[neighbour_node.y][neighbour_node.x].move_cost+lambda_*(gridmap_[neighbour_node.y][neighbour_node.x].heuristic-gridmap_[lowest_cost_node.y][lowest_cost_node.x].heuristic)
                if gridmap_[neighbour_node.y][neighbour_node.x].occupied:
                    # print("Occupied!")
                    continue
                if gridmap_[neighbour_node.y][neighbour_node.x].closed:
                    # print("Already closed, skipping.")
                    pass
                else:
                    # print("Contents of open_list:")
                    # for node in open_list:
                    #     node.print()
                    found, position = search_list_for_node(open_list, neighbour_node)
                    # for i in range (0, len(open_list)):
                    #     print(open_list[i])
                    if found:
                        # print("Already in list")
                        if neighbour_node.cost < open_list[position].cost:
                            open_list[position].cost = neighbour_node.cost
                            # print("Updating cost with lower cost found")
                    else:
                        # print("Not found, adding.")
                        open_list.append(Node(neighbour_node.x, neighbour_node.y, neighbour_node.cost))
                        gridmap_[neighbour_node.y][neighbour_node.x].expand(expansion)
            expansion += 1
            # print("")

    optimum_policy = []
    path_to_take = []
    check_node = Node()
    best_node = Node()
    current_node = Node(goal_node.x, goal_node.y, goal_node.cost)
    current_lowest_expansion = float("inf")
    optimum_policy.append(current_node)
    while True:
        image_map[current_node.y, current_node.x] = 255
        # cv.imshow("Image", image_map)
        # cv.waitKey(10)
        for d in directions:
            check_node.x = current_node.x+d.x
            check_node.y = current_node.y+d.y

            if gridmap_[check_node.y][check_node.x].expand_value < current_lowest_expansion:
                best_node = Node(check_node.x, check_node.y, check_node.cost)
                current_lowest_expansion = gridmap_[check_node.y][check_node.x].expand_value

        if start_node.x == best_node.x and start_node.y == best_node.y:
            break

        optimum_policy.append(Node(best_node.x, best_node.y, best_node.cost))
        current_node = Node(best_node.x, best_node.y, best_node.cost)

    for n in optimum_policy:
        path_to_take.append([n.x*factor, n.y*factor])

    return image_map, path_to_take

if __name__ == '__main__':
    image, optimum_policy_list = main(100,100,1000,200,(750,360))
    print(optimum_policy_list)
    image = cv2.resize(image, (1920, 1080))
    cv2.imshow("Image", image)
    cv2.waitKey()