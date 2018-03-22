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

    if(! is_initialized){
        default_random_engine gen;
        // Set the number of particles. Initialize all particles to first position (based on estimates of
        //   x, y, theta and their uncertainties from GPS) and all weights to 1.
        // Add random Gaussian noise to each particle.
        num_particles = 10;

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
        double x0, x1, y0, y1, theta0, theta1;

        Particle p_i = particles.at(i);

        x0 = p_i.x;
        y0 = p_i.y;
        theta0 = p_i.theta;

        if (fabs(yaw_rate) < 0.0001) {
            x1 = x0 + velocity  * cos(p_i.theta) * delta_t;
            y1 = y0 +  velocity  * sin(p_i.theta) * delta_t;
            theta1 = theta0 + 0.0001 * delta_t;
        } else {
            x1 = x0 +  velocity / yaw_rate * (sin(theta0 + yaw_rate * delta_t) - sin(theta0));
            y1 = y0 + velocity / yaw_rate * (cos(theta0) - cos(theta0 + yaw_rate * delta_t));
            theta1 = theta0 +  yaw_rate * delta_t;
        }

        normal_distribution<double> dist_x(x1, std_x);
        normal_distribution<double> dist_y(y1, std_y);
        normal_distribution<double> dist_theta(theta1, std_theta);

        p_i.x = dist_x(gen);
        p_i.y = dist_y(gen);
        p_i.theta = dist_theta(gen);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	//  Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
    int closest_predicted_index = 0;
    double min_dist = 99999999; // a very high value such that should be overwritten
    for (int i = 0; i< observations.size(); i++) {
        LandmarkObs curr_obs = observations.at(i);
        for (int j = 0; j< predicted.size(); j++) {
            LandmarkObs curr_predicted = predicted.at(j);
            double dist_curr = dist(curr_obs.x, curr_obs.y, curr_predicted.x, curr_predicted.y);
            if (dist_curr < min_dist) {
                min_dist = dist_curr;
                closest_predicted_index = j;
            }
        }
        curr_obs.id = closest_predicted_index;
    }

    cout << "Data association end"<< endl;

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    weights.clear();
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

//    vector<LandmarkObs> transformed_obs;
//    for (int j = 0; j < observations.size(); j++) {
//        LandmarkObs obs = observations.at(j);
//        LandmarkObs obs_trans;
//        double xm = p_i.x + cos(p_i.theta) * obs.x - obs.y * sin(p_i.theta);
//        double ym = p_i.y + sin(p_i.theta) * obs.x - obs.y * cos(p_i.theta);
//        obs_trans.x = xm;
//        obs_trans.y = ym;
//        transformed_obs.push_back(obs_trans);
//    }
    cout << "1: particles.size(): " << particles.size() << endl;

    for (int i = 0; i< particles.size(); i++) {
        Particle p_i = particles.at(i);

        vector<LandmarkObs> landmarksInRange;
        for (int a = 0; a < map_landmarks.landmark_list.size(); a++) {
            Map::single_landmark_s curr_landmark = map_landmarks.landmark_list.at(i);
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

        p_i.weight = 1.0;

        //the transformed_obs will each have their ids = the closest landmark's id
        dataAssociation(landmarksInRange, transformed_obs);
        cout <<"here 4" << endl;
        cout <<transformed_obs.size() << endl;
        // find the associated landmark by looping over the landmarks in range
        for (int k = 0; k< transformed_obs.size(); k++) {
            LandmarkObs curr_obst = transformed_obs.at(k);
            double xm = curr_obst.x;
            double ym = curr_obst.y;
            int curr_id = curr_obst.id;

            double mu_x;
            double mu_y;

            for (int l= 0; l< landmarksInRange.size(); l++) {
                if (landmarksInRange.at(l).id == curr_id) {
                    mu_x = landmarksInRange.at(l).x;
                    mu_y = landmarksInRange.at(l).y;
                }
            }
            cout << "asoociated"<< endl;
            double powx =  pow((xm - mu_x), 2) /(2 * pow(std_landmark_x, 2));
            double powy =  pow((ym - mu_y), 2) /(2 * pow(std_landmark_y, 2));

            p_i.weight *=  1 /(2 * M_PI * std_landmark_x * std_landmark_y ) * exp( -(powx + powy));
        }
//
////            <Map::single_landmark_s> minLandmark = map_landmarks.landmark_list.at(0);
//            double minDistance = dist(map_landmarks.landmark_list.at(0).x_f, map_landmarks.landmark_list.at(0).y_f, xm, ym);
//            float minX = map_landmarks.landmark_list.at(0).x_f;
//            float minY = map_landmarks.landmark_list.at(0).y_f;
//            int index = 0;
////            double minDistance = dist(minLandmark.x_f, minLandmark.y_f, xm, ym);
//
//            for  (int k = 1; k< map_landmarks.landmark_list.size(); k++ ){
////                <Map::single_landmark_s> currLandmark = map_landmarks.landmark_list.at(k);
//                double currDist = dist(map_landmarks.landmark_list.at(k).x_f, map_landmarks.landmark_list.at(k).y_f, xm, ym);
//                if (currDist < minDistance) {
//                    minDistance = currDist;
//                    index = k;
//                    minX = map_landmarks.landmark_list.at(k).x_f;
//                    minY = map_landmarks.landmark_list.at(k).y_f;
//                }
//            }
//
//            double powx =  pow((xm - minX), 2) /(2 * pow(std_landmark[0], 2));
//            double powy =  pow((ym - minY), 2) /(2 * pow(std_landmark[1], 2));
//
////            double powx =  pow((xm - minLandmark.x_f), 2) /(2 * pow(std_landmark[0], 2));
////            double powy =  pow((ym - minLandmark.y_f), 2) /(2 * pow(std_landmark[1], 2));
//
//            p_i.weight = p_i.weight*  1 /(2 * M_PI * std_landmark[0] * std_landmark[1] ) * pow(M_E, -(powx + powy));
//
//
//
//        }
        weights.push_back(p_i.weight);
//        particles.at(i) = p_i;
//
//        cout << "new weight"<< weights.at(i)<< endl;
//        if (weights.at(i) == 1) {
//            cout << "pi weidght:"<< i<< ","<< p_i.weight << endl;
//        }
//        cout << "particles" << particles.at(i).weight<<endl;
    }


}




void ParticleFilter::resample() {
    default_random_engine gen;
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    discrete_distribution<int> dist_weights(weights.begin(), weights.end());
    vector<Particle> resampled_particles;
    for (int i = 0; i< particles.size() ; i++) {
        int index = dist_weights(gen);
        cout << "index: "<< index<< endl;
        Particle chosen = particles.at(dist_weights(gen));
        resampled_particles.push_back(chosen);
    }
    particles = resampled_particles;

    int num_particles = particles.size();
    double highest_weight = -1.0;
    Particle best_particle;
    double weight_sum = 0.0;
    for (int i = 0; i < num_particles; ++i) {
        if (particles[i].weight > highest_weight) {
            highest_weight = particles[i].weight;
            best_particle = particles[i];
        }
        weight_sum += particles[i].weight;
    }
    cout << "highest w " << highest_weight << endl;
    cout << "average w " << weight_sum/num_particles << endl;
    cout << "position" << best_particle.x<<","<<best_particle.y<< endl;
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
