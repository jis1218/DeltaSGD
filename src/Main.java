
public class Main {
	public static void main(String args[]){
		
		double[][] sample = {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}};
		double[] weight = {0.5, 0.5, 0.5};
		double[] target = {0.3, 0.2, 0.5, 0.6};
		SGDDelta sgdDelta = new SGDDelta();
		
		for(int m=0; m<100000; m++){
			for(int i=0; i<sample.length; i++){
				weight = sgdDelta.DoSGDDelta(sample[i], weight, target[i]);
			}
		}
		
		double output[] = new double[sample.length];
		
		// output 출력하는 함수
		for(int k=0; k<sample.length; k++){
			double sum = 0;
			for(int j=0; j<weight.length; j++){
				sum += sample[k][j]*weight[j];
			}
			output[k] = sgdDelta.sigmoid(sum);
			System.out.println(output[k]);
		}	
	}

}
