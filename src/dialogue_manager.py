import random
import itertools

import numpy as np
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer

class Manager:
    def __init__(self, dis_sym):
        self.dis_sym = dis_sym
        self.diseases = list(dis_sym.keys())
        symptoms = [v for k,v in dis_sym.items()]
        self.symptoms = list(set(itertools.chain.from_iterable(symptoms)))
        self.curr_disease = None
        self.curr_disease_symptoms = None
        self.curr_disease_symptoms_asked = None
        self.dis_sym_enc = OneHotEncoder().fit(np.array(self.diseases + self.symptoms).reshape(-1,1))
        self.sym_enc = MultiLabelBinarizer().fit(np.array(self.symptoms).reshape(-1, 1))
        
    def reset(self):
        # print()
        self.curr_disease = random.choice(self.diseases)
        self.curr_disease_symptoms = self.dis_sym[self.curr_disease]
        first_symptoms = random.sample(self.curr_disease_symptoms, 4)
        # print(first_symptoms)
        self.curr_disease_symptoms_asked = first_symptoms
        return self.sym_enc.transform([self.curr_disease_symptoms_asked])
    
    def step(self, action):
        action = self.dis_sym_enc.categories_[0][action]
        if action == self.curr_disease:
            done = True
            reward = 50  # For detecting right disease
            observation_ = self.curr_disease_symptoms_asked
        elif action in self.diseases:
            done = True
            reward = 0 #For detecting wrong disease
            observation_ = self.curr_disease_symptoms_asked
        elif action in self.curr_disease_symptoms:
            done = False
            reward = 1 #For asking right question
            if action not in self.curr_disease_symptoms_asked:
                self.curr_disease_symptoms_asked.append(action)
                observation_ = self.curr_disease_symptoms_asked
        else:
            done = False
            reward = 0 #For wasting question
            observation_ = self.curr_disease_symptoms_asked
        
        # print(len(self.curr_disease_symptoms_asked))
        observation_ = self.sym_enc.transform([self.curr_disease_symptoms_asked])
        info = ''
            
        return observation_, reward, done, info
