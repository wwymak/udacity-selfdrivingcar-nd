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
#include "map.h"

using namespace std;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
    if(is_initialized == false){

        default_random_engine gen;
        // Set the number of particles. Initialize all particles to first position (based on estimates of
        //   x, y, theta and their uncertainties from GPS) and all weights to 1.
        // Add random Gaussian noise to each particle.
        num_particles = 100;

        double std_x = std[0];
        double std_y = std[1];
        double std_theta = std[2];

        normal_distribution<double> dist_x(x, std_x);
        normal_distribution<double> dist_y(y, std_y);
        normal_distribution<double> dist_theta(theta, std_theta);

        for(int i = 0; i< num_particles; i++) {
            Particle p;
            p.id = i;
            p.x = dist_x(gen);
            p.y = dist_y(gen);
            p.theta = dist_theta(gen);
            p.weight = 1.0;

            particles.push_back(p);
            weights.push_back(1.0);
        }

        is_initialized = true;
    }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// update the position of each each particle and add random Gaussian noise.
    default_random_engine gen;

    double std_x = std_pos[0];
    double std_y = std_pos[1];
    double std_theta = std_pos[2];

    for (int i = 0; i< num_particles; i++) {

        if (fabs(yaw_rate) < 0.00001) {
            particles.at(i).x += velocity  * cos(particles.at(i).theta) * delta_t;
            particles.at(i).y += velocity  * sin(particles.at(i).theta) * delta_t;
            particles.at(i).theta += 0.00001 * delta_t;
        } else {
            particles.at(i).x +=  velocity / yaw_rate * (sin(particles.at(i).theta + yaw_rate * delta_t) - sin(particles.at(i).theta));
            particles.at(i).y += velocity / yaw_rate * (cos(particles.at(i).theta) - cos(particles.at(i).theta + yaw_rate * delta_t));
            particles.at(i).theta +=  yaw_rate * delta_t;
        }

        normal_distribution<double> dist_x(particles.at(i).x, std_x);
        normal_distribution<double> dist_y(particles.at(i).y, std_y);
        normal_distribution<double> dist_theta(particles.at(i).theta, std_theta);

        particles.at(i).x = dist_x(gen);
        particles.at(i).y = dist_y(gen);
        particles.at(i).theta = dist_theta(gen);
    }

}

//void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
//    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
//    //   observed measurement to this particular landmark.
//    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
//    //   implement this method and use it as a helper during the updateWeights phase.
//    // Source Forum help: https://discussions.udacity.com/t/you-ran-out-of-time-when-running-with-the-simulator/269900/8
//
//    double distance, min_distance;
//    int map_index;
//
//    for (unsigned int i = 0; i < observations.size(); i++)
//    {
//        LandmarkObs observation = observations[i];
//        // Thanks to Jeremy Shanoon for teaching me about numeric_limits
//        min_distance = 99999999;
////        min_distance = numeric_limits<double>::max();
//
//        for (unsigned int j = 0; j < predicted.size(); j++)
//        {
//            LandmarkObs l = predicted[j];
//            distance = dist(observation.x, observation.y, l.x, l.y);
//
//            if (distance < min_distance)
//            {
//                min_distance = distance;
//                map_index = l.id;
//            }
//        }
//        observations[i].id = map_index;
//    }
//
//}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	//  Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.

    int associated_id;
    double min_dist;
    double dist_curr;

    for (int i = 0; i< observations.size(); i++) {
        LandmarkObs curr_obs = observations.at(i);
        min_dist = 99999999;
        for (int j = 0; j< predicted.size(); j++) {
            LandmarkObs curr_predicted = predicted.at(j);
            dist_curr = dist(curr_obs.x, curr_obs.y, curr_predicted.x, curr_predicted.y);
            if (dist_curr < min_dist) {
                min_dist = dist_curr;
                associated_id = curr_predicted.id;
            }
        }
        observations.at(i).id = associated_id;
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], std::vector<LandmarkObs> &observations,
                                   Map &map_landmarks) {
//    weights.clear();
    const double std_landmark_x = std_landmark[0];
    const double std_landmark_y = std_landmark[1];
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about thias distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    for (int i = 0; i< particles.size(); i++) {
        Particle& p_i = particles.at(i);

        vector<LandmarkObs> landmarksInRange;
        for (int a = 0; a < map_landmarks.landmark_list.size(); a++) {
            Map::single_landmark_s curr_landmark = map_landmarks.landmark_list.at(a);
            double distToMark = dist(curr_landmark.x_f, curr_landmark.y_f, p_i.x, p_i.y);
            if (distToMark <= sensor_range) {
                LandmarkObs obs_in_range;
                obs_in_range.id = curr_landmark.id_i;
                obs_in_range.x = curr_landmark.x_f;
                obs_in_range.y =  curr_landmark.y_f;
                landmarksInRange.push_back(obs_in_range);
            }
        }

        vector<LandmarkObs> transformed_obs;
        for (int j = 0; j < observations.size(); j++) {
            LandmarkObs obs = observations.at(j);

            LandmarkObs obs_trans;
            obs_trans.id = obs.id;
            double xm = p_i.x + cos(p_i.theta) * obs.x - obs.y * sin(p_i.theta);
            double ym = p_i.y + sin(p_i.theta) * obs.x + obs.y * cos(p_i.theta);
            obs_trans.x = xm;
            obs_trans.y = ym;
            transformed_obs.push_back(obs_trans);
        }

        //the transformed_obs will each have their ids = the closest landmark's id
        dataAssociation(landmarksInRange, transformed_obs);

        p_i.weight = 1.0;
        // find the associated landmark by looping over the landmarks in range
        for (int k = 0; k< transformed_obs.size(); k++) {
            LandmarkObs curr_obst = transformed_obs.at(k);
            double xm = curr_obst.x;
            double ym = curr_obst.y;
            int curr_id = curr_obst.id;

            double mu_x;
            double mu_y;
            int assoc_id;

            for (int l= 0; l< landmarksInRange.size(); l++) {
                if (landmarksInRange.at(l).id == curr_id) {
                    mu_x = landmarksInRange.at(l).x;
                    mu_y = landmarksInRange.at(l).y;
                    assoc_id = landmarksInRange.at(l).id;
                }
            }

            double powx =  pow((xm - mu_x), 2) /(2 * pow(std_landmark_x, 2));
            double powy =  pow((ym - mu_y), 2) /(2 * pow(std_landmark_y, 2));

            p_i.weight *=  1 /(2 * M_PI * std_landmark_x * std_landmark_y ) * exp( -(powx + powy));
        }

        weights[i] = p_i.weight;
    }


}


void ParticleFilter::resample() {
    default_random_engine gen;
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    discrete_distribution<int> dist_weights(weights.begin(), weights.end());
    vector<Particle> resampled_particles;
    for (int i = 0; i< num_particles ; i++) {
        int index = dist_weights(gen);
        resampled_particles.push_back(particles.at(dist_weights(gen)));
    }
    particles = resampled_particles;
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

    return particle;
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
