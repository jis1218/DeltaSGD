```java
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
```
```java
public static void main(String args[]){
		
		double[][] sample = {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}};
		double[] weight = {0.5, 0.5, 0.5};
		double[] target = {0, 0, 1, 1};
		SGDDelta sgdDelta = new SGDDelta();
		
		for(int m=0; m<10000; m++){
			for(int i=0; i<sample.length; i++){
				weight = sgdDelta.DoSGDDelta(sample[i], weight, target[i]);
			}
		}
```

##### 위의 결과를 epoch=10000으로 하였을 때의 결과
##### 0.010198540628012683, 0.008295185088533857, 0.9932409356415381, 0.991687093153866

##### 하지만 target값을 0,0,1,1이 아닌 다른 값으로 주었을 때... 예를 들어 0.3, 0.2, 0.5, 0.6 으로 주게 되면 epoch를 100000으로 하여도 위와 같이 근사치를 얻기가 힘들다. (이렇게 나온다. 0.24047460914164168, 0.25508093679021554, 0.5438251236277521, 0.5631942629727512) 그 이유는 무엇이며 그것이 뜻하는 바가 무엇인가?

##### 궁금증 2가지
##### 1. 은닉을 왜 하는 것인가?

##### 2. Actual function으로 특정함수(Sigmoid 함수 등)을 쓰는 이유는 무엇인가?