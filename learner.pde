class Learner{
	int DIMENSIONS = 5;
	int[] BUCKETS = {10,10,10,10,10};

	float[][] Q = new float[DIMENSIONS][2];
	float epsilon,discount;

	Learner(float epsilon, float discount){
		this.epsilon = epsilon;
		this.discount = discount;
	}

	void viewWorld(State s){

	}

	void act(){

	}

	void learn(){

	}
}
