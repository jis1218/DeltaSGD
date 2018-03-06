
public class Batch {
	public double[] doBatch(double[][] sample, double[] weight, double[] target){
		
		double alpha = 0.9;
		double weight_sum = 0;
		double delta = 0;
		double error = 0;
		double deltaW[] = new double[weight.length];
		double output = 0;
		
		for(int i=0; i<sample.length; i++){			
			for(int j=0; j<weight.length; j++){
				weight_sum += sample[i][j]*weight[j];
				error = target[i]-sigmoid(weight_sum);
				output = sigmoid(weight_sum);
				delta = output*(1-output)*error;	
				deltaW[j] +=alpha*delta*sample[i][j];
				weight[j] += deltaW[j]/sample.length; // deltaW의 평균을 weight에 더해준다.
			}		
		}		
		
		return weight;	
			
	}
	
	public double sigmoid(double v){		
		return 1/(1+Math.exp(-1*v));	
	}

}
