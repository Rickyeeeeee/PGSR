/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define NUM_ALL_MAP 5
#define ALL_MAP_NORMAL_X 0
#define ALL_MAP_NORMAL_Y 1
#define ALL_MAP_NORMAL_Z 2
#define ALL_MAP_ONE 3
#define ALL_MAP_DISTANCE 4
#define BLOCK_X 16
#define BLOCK_Y 16

#endif