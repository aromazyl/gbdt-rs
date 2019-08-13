//
// tree.rs
// Copyright (C) 2019 zhangyule <zyl2336709@gmail.com>
// Distributed under terms of the MIT license.
//

#[derive(Copy, Clone, Debug)]
struct TreeStatsNode {
    s: f64,
    min_delta_loss: f64,
    fea_value: f64,
    m_s: f64,
    l_s: f64,
}

#[derive(Copy, Clone, Debug)]
struct TreeNode {
    left_child_index: i32,
    right_child_index: i32,
    feature_id: i32,
    split_value: f64,
    label: f32,
    m_inf: i32,
    l_inf: f64,
    stats: Vec<TreeStatsNode>,
    gain: f64,
}

#[derive(Debug)]
struct Tree {
    depth: i32,
    node_num: i32,
    doc_num: i32,
    fea_num: i32,
    nodes: Vec<TreeNode>,
    node_idx: Vec<i32>,
    fx: Vec<f64>,
    g: Vec<f64>,
    fea_significace: Vec<f64>,
}

trait TreeMethod {
    fn new(depth: i32, num_docs: i32, num_fea: i32) -> Self;
    fn reset(&mut self);
    fn initFx(&mut self, data: &Data);
    fn updateFx(&mut self, data: &Data);
    fn calculateGradient(&mut self, data: &Data);
    fn calculateCoefficient(&mut self, data: &Data);
}
