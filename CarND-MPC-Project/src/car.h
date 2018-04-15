//
// Created by Wing Yee mak on 15/04/2018.
//

#ifndef MPC_CAR_H
#define MPC_CAR_H


class car {
public:

    double px; //position x
    double py; //position y
    double v; //velocity
    double psi; //car angle
    double cte;
    double steer;
    double acc;
    /**
   * Constructor
   */
    car();

    /**
   * Destructor
   */
    virtual ~car();

    void Drive(const double dt);


};


#endif //MPC_CAR_H
