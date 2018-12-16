
public class Main {
	public static void main(String args[]){
		
		double[][] sample = {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}};
		double[] weight = {0.5, 0.5, 0.5};
		double[] target = {0, 0, 1, 1};
		SGDDelta sgdDelta = new SGDDelta();
		Batch batch = new Batch();
		
		for(int m=0; m<100000; m++){
			for(int i=0; i<sample.length; i++){
				//DoSGDDelta �Լ��� ���� �Ѱ��� �ѹ��� weight�� �����Ѵ�.
				weight = sgdDelta.DoSGDDelta(sample[i], weight, target[i]);
			}
		}
		
		// output�� ���� �迭 ����
		double output[] = new double[sample.length];
		
		// output ����ϴ� �Լ�
		for(int k=0; k<sample.length; k++){
			double sum = 0;
			for(int j=0; j<weight.length; j++){
				sum += sample[k][j]*weight[j];
			}
			output[k] = sgdDelta.sigmoid(sum);
			System.out.println(output[k]);
		}
		
		
		double batch_output = 0;
		
		for(int z=0; z<100000; z++){
			// batch ������� ��� �����غ���, ���� ��Ʈ�� �ѹ��� weight�� �����Ѵ�.
			weight = batch.doBatch(sample, weight, target);
		}
		
		// output ����ϴ� �Լ�
		for(int w=0; w<sample.length; w++){
			double output_sum = 0;
			for(int y=0; y<weight.length; y++){
				output_sum += sample[w][y]*weight[y];
			}
			batch_output = batch.sigmoid(output_sum);
			System.out.println(batch_output);
		}
	}
}
