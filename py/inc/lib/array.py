#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###

class MS_Array:

    @staticmethod
    def reorder_edges_to_center(arr):
        result = []
        left = 0
        right = len(arr) - 1
        
        while left <= right:
            if left == right:
                result.append(arr[left])
            else:
                result.append(arr[left])
                result.append(arr[right])
            left += 1
            right -= 1
        
        return result
