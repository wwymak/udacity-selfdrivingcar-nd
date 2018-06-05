//
// Created by Wing Yee mak on 21/05/2018.
//

#include "prediction.h"

vector <vector<double>> predictNewState(float time) {

}

void updateState(vector<float>othercars_s, vector<float>othercars_d, vector<float>othercars_v, float time) {
    vector<vector<float>>state;
    for (int i = 0; i< othercars_s.size(); i++) {
        float new_s = othercars_s.at(i) + othercars_v.at(i) * time;
        state.push_back({new_s, othercars_d.at(i)});
    }
}