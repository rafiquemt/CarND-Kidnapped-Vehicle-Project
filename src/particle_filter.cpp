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
#include <limits>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	num_particles = 100;
	particles.resize(num_particles);
	weights.resize(num_particles);
	double w = 1;
	
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	for (int i = 0; i < particles.size(); i++) {
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = w;
		particles[i] = p;
		weights[i] = 1;
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	for (int i = 0; i < particles.size(); i++) {
		Particle& p = particles[i];
		double x, y, theta;
		
		if (abs(yaw_rate) < 0.001) {
			x = p.x + velocity * delta_t * cos(p.theta);
			y = p.y + velocity * delta_t * sin(p.theta);
			theta = p.theta;
		} else {
			x = p.x + (velocity / yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
			y = p.y + (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
			theta = p.theta + yaw_rate * delta_t;
		}
		theta = normalizeToPi(theta);

		normal_distribution<double> dist_x(x, std_pos[0]);
		normal_distribution<double> dist_y(y, std_pos[1]);
		normal_distribution<double> dist_theta(theta, std_pos[2]);
		// add gaussian noise and update particle
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = normalizeToPi(dist_theta(gen));
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	double std_lx = std_landmark[0];
	double std_ly = std_landmark[1];
	auto actualLandmarks = map_landmarks.landmark_list;
	// Transform observations from vehicle to map coordinates
	for (int i = 0; i < particles.size(); i++) {
		auto obs_map_list = std::vector<LandmarkObs>(observations.size());
		vector<int> associations;
		vector<double> sense_x;
		vector<double> sense_y;
		Particle& p = particles[i];
		double x = p.x;
		double y = p.y;
		double theta = p.theta;
		
		double weight = 1;

		vector<LandmarkObs> predictedLandmarks;
		for(auto map_landmark : map_landmarks.landmark_list) {
			LandmarkObs predicted {map_landmark.id_i, map_landmark.x_f, map_landmark.y_f};
			if (dist(p.x, p.y, predicted.x, predicted.y) <= sensor_range) {
				predictedLandmarks.push_back(predicted);
			}
		}

		for (int j = 0; j < observations.size(); j++) {
			LandmarkObs &obs = observations[j];
			LandmarkObs obs_map;
			obs_map.x = x + obs.x * cos(theta) - obs.y * sin(theta);
			obs_map.y = y + obs.x * sin(theta) + obs.y * cos(theta);
			obs_map_list[j] = obs_map;

			// find landmark that this observation is closest to
			double bestResult = numeric_limits<double>::max();
			double associationIndex = -1; // not found
			
			for (int k = 0; k < predictedLandmarks.size(); k++) {
				LandmarkObs predicted = predictedLandmarks[k];
				double offset = dist(obs_map.x, obs_map.y, predicted.x, predicted.y);
				if (offset < bestResult) {
					bestResult = offset;
					associationIndex = predicted.id;
				}
			}
			auto matchingLandmark = actualLandmarks[associationIndex - 1];
			weight = weight * gaussian_prob(obs_map.x, obs_map.y, matchingLandmark.x_f, matchingLandmark.y_f, std_lx, std_ly);
			associations.push_back(associationIndex); // + 1 because the landmark ids start from 1. 
			sense_x.push_back(obs_map.x);
			sense_y.push_back(obs_map.y);
		}
		p.weight = weight; // transform from log
		weights[i] = p.weight;
		SetAssociations(p, associations, sense_x, sense_y);
	}

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
}

void ParticleFilter::resample() {
	return;
	discrete_distribution<int> dist(weights.begin(), weights.end());
	int n = num_particles;
	vector<Particle> resampled(n);
	for (int i = 0; i < n; i++) {
		int index = dist(gen);
		cout << endl << index << "t" << endl;
		Particle& p = particles[index];
		resampled.push_back(p);
	}
	particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

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
