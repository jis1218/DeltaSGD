
public class SGDDelta {
	
	public double[] DoSGDDelta(double[] sample, double[] weight, double target){
		double alpha = 0.9;
		double output = 0;
		double error = 0;
		double delta = 0;
		double deltaW = 0;
		double weighted_sum = 0;
		
		for(int i=0; i<weight.length; i++){			
			weighted_sum += sample[i]*weight[i]; //�ռ����� ���Ѵ�.
		}
		
		output = sigmoid(weighted_sum); // sigmoid �Լ��� �־� actual output�� ���Ѵ�.
		error = target - output; // target output�� actual output�� ���� error�� ���Ѵ�.
		delta = output*(1-output)*error;
		
		// ���� deltaWeight ���� �̿��Ͽ� weight�� �����Ѵ�.
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
