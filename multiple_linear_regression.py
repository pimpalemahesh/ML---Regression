import numpy as np

class multiple_linear_regression():

    def __init__(self, X, Y):

        self.input = np.array(X)
        self.output = np.array(Y)

        new_inputs = np.insert(self.input, 0, 1, axis=1)
        
        self.theta = np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(new_inputs),new_inputs)), np.transpose(new_inputs)), self.output)
        print(self.theta)
        
    def calculate(self, x1, x2):
        answer = self.theta[0] + self.theta[1]*x1 + self.theta[2]*x2
        print(round(answer))

    def calculate_for_all(self, input):
        output = []
        for i in range(len(input)):
            output.append(round(self.theta[0] + self.theta[1]*input[i][0] + self.theta[2]*input[i][1]))
        return output

    def root_mean_error(self, o_desired, o_predicted):
        n = len(o_desired)
        total = 0.0
        for i in range(n):
            total += (o_desired[i] - o_predicted[i])**2
        return total/n

inputs = [[1,2],[2,3],[3,1],[4,5],[5,4],[6,3],[7,6],[8,4],[9,8]]
outputs = [3,4,6,8,10,11,12,15,16]

mlr = multiple_linear_regression(inputs, outputs)
output_predicted = mlr.calculate_for_all(inputs)
print("Output List :")
print(output_predicted)
print("Root Mean square Error : " )
print(mlr.root_mean_error(outputs, output_predicted))