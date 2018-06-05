//
// Where are all the other cars at time t later
//

#ifndef PATH_PLANNING_PREDICTION_H
#define PATH_PLANNING_PREDICTION_H

#include <vector>

using namespace std;
class Prediction {
private:
    vector<float> othercars_s;
    vector<float> othercars_d;
    vector<float> othercars_v;

    void updateState(vector<float>othercars_s, vector<float>othercars_d, vector<float>othercars_v, float time);

public:
    bool tooClose(float s, float d, float v, float time);
//    bool availableSpace()
};


#endif //PATH_PLANNING_PREDICTION_H
