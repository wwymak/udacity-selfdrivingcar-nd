/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
default_random_engine gen;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
    int num_particles = 100;

    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for(int i = 0; i< num_particles; i++) {
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1;

        particles.push_back(p);
        weights.push_back(1);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// update the position of each each particle and add random Gaussian noise.


    for (int i = 0; i< particles.size(); i++) {
        Particle p_i = particles.at(i);

        if (yaw_rate == 0) {
            p_i.x += velocity  * cos(p_i.theta) * delta_t;
            p_i.y += velocity  * sin(p_i.theta) * delta_t;
        } else {
            p_i.x += velocity / yaw_rate * (sin(p_i.theta + yaw_rate * delta_t) - sin(p_i.theta));
            p_i.y += velocity / yaw_rate * (cos(p_i.theta) - cos(p_i.theta + yaw_rate * delta_t));
            p_i.theta += yaw_rate * delta_t;
        }

        normal_distribution<double> dist_x(p_i.x, std_pos[0]);
        normal_distribution<double> dist_y(p_i.y, std_pos[1]);
        normal_distribution<double> dist_theta(p_i.theta, std_pos[2]);

        p_i.x = dist_x(gen);
        p_i.y = dist_y(gen);
        p_i.theta = dist_theta(gen);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
//    float distOar
    int closest_predicted_index = 0;
    int min_dist;
    for (int i = 0; i< observations.size(); i++) {
        for (int j = 0; j< predicted.size(); j++) {
            double dist_curr = dist(observations.at(i).x, observations.at(i).y, predicted.at(j).x, predicted.at(j).y);
            if(j == 0) {
                min_dist = dist_curr;
            } else if (dist_curr < min_dist) {
                min_dist = dist_curr;
                closest_predicted_index = j;
            }
        }

    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    for (int i = 0; i< particles.size(); i++) {
        Particle p_i = particles.at(i);
        vector<LandmarkObs> transformed_obs;
        for (int j = 0; j < observations.size(); j++) {
            LandmarkObs obs = observations.at(j);
            LandmarkObs obs_trans;
            double xm = p_i.x + cos(p_i.theta) * obs.x - obs.y * sin(p_i.theta);
            double ym = p_i.y + sin(p_i.theta) * obs.x - obs.y * cos(p_i.theta);
            obs_trans.x = xm;
            obs_trans.y = ym;
            transformed_obs.push_back(obs_trans);
        }

        double a = p_i.x -

        double P = 1 / (2 * M_PI * std_landmark[0], std_landmark[1]) * exp()
    }

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    discrete_distribution<int> dist_weights(weights.begin(), weights.end());
    vector<Particle> resampled;
    for (int i = 0; i< particles.size(); i++) {
        resampled.push_back(particles.at(dist_weights(gen));
    }
    particles = resampled;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
