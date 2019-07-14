#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <thread>
#include "layer.h"



std::vector<int> classes={};

struct Sample{
    std::vector<float> input;
    int expected;
    Sample(std::vector<float> i, int e){
        input=i;
        expected=e;
    }
};
//legacy code 
struct Network{
    std::vector<std::vector<Node>> model;
    std::vector<Sample> dataset;
    std::vector<int> config;
    Network(){   
    }
    float activation(float activation){
        return 1/(1+exp(-activation));
    }
    float derivative(float out){
        return out*(1-out);
    }
    void print(std::vector<std::vector<Node>>& clone){
        for(int x=0;x<clone.size();x++){
            for(int y=0;y<clone[x].size();y++){
                std::cout << "[A: " << clone[x][y].activation << ", D:" << clone[x][y].delta << ", B: " << clone[x][y].bias << "] ";
            }
            std::cout << std::endl;
            for(int y=0;x!=clone.size()-1&&y<clone[x].size();y++){
                std::cout << "[[ ";
                for(int k=0;k<clone[x][y].weights.size();k++){
                    std::cout << clone[x][y].weights[k] << " ";
                }
				std::cout << "]]" << " ";
            }
            std::cout << std::endl;
        }
    }
    void init(std::vector<Sample> samples,std::vector<int> conf){
        config=conf;
        config.insert(config.begin(),samples[0].input.size());
        config.push_back(classes.size());
        for(int x=0;x<config.size();x++){
            model.push_back({});
            for(int y=0;y<config[x];y++){
                model[x].push_back(Node());
                if(x!=config.size()-1){
                    model[x][y].weights=std::vector<float>(config[x+1]);
                    for(int k=0;k<model[x][y].weights.size();k++){
                        model[x][y].weights[k]=float(rand()%200)/100-1;
                    }
                }
            }
        }
        dataset=samples;
    }
    void propagate(std::vector<std::vector<Node>>& clone,Sample sample){
        for(int y=0;y<clone[0].size();y++){
            clone[0][y].value=sample.input[y];
        }
        for(int x=0;x<clone.size();x++){
            for(int y=0;y<clone[x].size();y++){
                clone[x][y].activation=activation(clone[x][y].value+clone[x][y].bias);
                for(int k=0;k<clone[x][y].weights.size();k++){
                    clone[x+1][k].value+=clone[x][y].weights[k]*clone[x][y].activation;
                }
            }
        }
    }
    void backprop(std::vector<std::vector<Node>>& clone){
        for(int x=clone.size()-2;x>=0;x--){
            for(int y=0;y<clone[x].size();y++){
                float total=0;
                for(int k=0;k<clone[x][y].weights.size();k++){
                    total+=clone[x][y].weights[k]*clone[x+1][k].delta;
                }
                clone[x][y].delta=total*derivative(clone[x][y].activation);
            }
        }
    }
    void update(std::vector<std::vector<Node>>& clone){
        for(int x=clone.size()-2;x>=0;x--){
            for(int y=0;y<clone[x].size();y++){
                for(int k=0;k<clone[x][y].weights.size();k++){
                    model[x][y].weights[k]+=0.01*clone[x+1][k].delta*clone[x][y].activation;
                }
            }
        }
        for(int x=clone.size()-1;x>0;x--){
            for(int y=0;y<clone[x].size();y++){
                model[x][y].bias+=0.01*clone[x][y].delta;
            }
        }
    }
    float cost(std::vector<std::vector<Node>>& clone,Sample sample){
        float total=0;
        for(int i=0;i<clone[clone.size()-1].size();i++){
            total+=0.5*pow((sample.expected==i?1:0)-clone[clone.size()-1][i].activation,2);
            clone[clone.size()-1][i].delta=(sample.expected==i?1:0)-clone[clone.size()-1][i].activation;
        }
        return total;
    }
    void train(float rate, float minError){
        float totalError=INFINITY;
        int i;
        for(i=0;minError<totalError;i++){
            totalError=0;
            for(int k=0;k<dataset.size();k++){
                std::vector<std::vector<Node>> clone=model;
                propagate(clone,dataset[k]);
                float caseCost=cost(clone,dataset[k]);
                backprop(clone);
                update(clone);
                totalError+=caseCost;
            }
            std::cout << "Epoch: " << i+1 << ", Total Error: " << totalError << std::endl;
        }
        std::cout << "Finished training in " << i << " epochs" << std::endl;
        print(model);
    }
};

int main(){
    Network network=Network();
    std::vector<Sample> samples;
    std::set<int> classSet;
    for(int i=0;i<50;i++){
        int n=5;
        int a=rand()%n;
        int b=rand()%n;
        int c=(a|b);
        Sample sample=Sample({float(a-(float(n)/2)),float(b-(float(n)/2))},c);
        samples.push_back(sample);
    }
    for(int i=0;i<samples.size();i++){
        classSet.insert(samples[i].expected);
    }
    classes=std::vector<int>(classSet.size());
    copy(classSet.begin(),classSet.end(),classes.begin());
    for(int i=0;i<samples.size();i++){
        samples[i].expected=distance(classes.begin(),find(classes.begin(),classes.end(),samples[i].expected));
    }
    network.init(samples,{5,5});
    network.train(0.1,0.001);
    return 0;
}
