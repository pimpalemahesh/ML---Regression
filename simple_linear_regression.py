import numpy as np

class Simple_Linear_Regression():

    """
        Class to calculate simple regression
    """

    def __init__(self, X, Y):

        """
            Constructor:
                Args :
                X (list) : input value of feature
                Y (list) : corresponding output values of feature.

        """
        self.input = np.array(X)
        self.output = np.array(Y)

        self.mean_x = sum(self.input)/len(self.input)
        self.mean_y = sum(self.output)/len(self.output)

        self.mean_sq_x = np.dot(self.input, self.input)/len(self.input)
        self.mean_xy = np.dot(self.input, self.output)/len(self.input)

        self.a1 = (self.mean_xy - self.mean_x*self.mean_y)/((self.mean_sq_x) - (self.mean_x)**2)

        self.a0 = self.mean_y - self.a1*self.mean_x

    def calculate_for_all(self, x_input):
        """
        Main Method:
            Args:
            x (float) : value of feature
            Return:
            output value of class y(float)
        """
        
        lst = [round(self.a1*x + self.a0, 2) for x in x_input]
        return lst

    def calculate_for_one(self, x):
        return round(self.a1*x + self.a0, 2)
        
    def root_mean_error(self, o_desired, o_predicted):
        n = len(o_desired)
        total = 0.0
        for i in range(n):
            total += (o_desired[i] - o_predicted[i])**2
        return total/n

inputs = [6.2, 6.5, 5.4, 6.5, 7.1, 7.9, 8.5, 8.9, 9.5, 10.6]
outputs = [26.3, 26.6, 25, 26, 27.9, 30.4, 35.4, 38.5, 42.6, 48.3]


x = 10.6
slr = Simple_Linear_Regression(inputs, outputs)
ot = slr.calculate_for_all(inputs)
print("My outputs : ")
print(ot)
error = slr.root_mean_error(outputs, ot)
print("Root Mean square error : ",error)