
public class SGDDelta {
	
	public double[] DoSGDDelta(double[] sample, double[] weight, double target){
		double alpha = 0.9;
		double output = 0;
		double error = 0;
		double delta = 0;
		double deltaW = 0;
		double weighted_sum = 0;
		
		for(int i=0; i<weight.length; i++){			
			weighted_sum += sample[i]*weight[i]; //합성곱을 구한다.
		}
		
		output = sigmoid(weighted_sum); // sigmoid 함수에 넣어 actual output을 구한다.
		error = target - output; // target output과 actual output을 빼어 error를 구한다.
		delta = output*(1-output)*error;
		
		// 얻은 deltaWeight 값을 이용하여 weight을 갱신한다.
		for(int j=0; j<weight.length; j++){
			deltaW = alpha*delta*sample[j];			
			weight[j] = weight[j] + deltaW;	
		}	
		
		return weight;
	}
	
	public double sigmoid(double v){		
		return 1/(1+Math.exp(-1*v));	
	}
}
