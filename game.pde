enum PhysicsModel{
	MOVE, JUMP, HOVER
}

enum Actions{
	UP, DOWN
}

class FlappyBird {
	PhysicsModel world;
	ArrayList<Obstacle> obstacles = new ArrayList<Obstacle>();

	int obstacleGap = 50;

	FlappyBird(PhysicsModel w){
		world = w;
		for(int i = 0; i < stageWidth % Obstacle.width; i++){
			
		}
	}

	public void nextFrame(){

	}

	public void takeAction(Action a){
		switch(a){
			case UP:
				jump();
				break;
			case DOWN:
				fall();
				break;
		}
	}

	public State currentState(){
	
	}

	// internal game methods

	private void addObstacleAt(int x){

	}

	private void jump(){
		switch(world){
			case MOVE:
			case JUMP:
			case HOVER:
		}
	}

	private void fall(){
		switch(world){
			case MOVE:
			case JUMP:
			case HOVER:
		}
	}
}

class State{
	float[] coordinates;
	
	State(){
		coordinates = new float[DIMENSIONS];
	}

	State(float[] m){
		coordinates = m;
	}

	void setDimension(int i, float x){
		coordinates[i] = x;
	}

	float getDimension(int i){
		return coordinates[i];
	}
}

class Obstacle{
	static const width = 10;
}

// game utilities 

// intersectAABB : check axis-aligned bounding boxes (AABB) for intersection

boolean intersectAABB(float x1, float y1, float w1, float h1, float x2, float y2, float w2, float h2) {
  float left1 = x1;
  float left2 = x2;
  float right1 = x1 + w1;
  float right2 = x2 + w2;

  float top1 = y1;
  float top2 = y2;
  float btm1 = y1 + h1;
  float btm2 = y2 + h2;

  boolean xhit = false;
  boolean yhit = false;

  // overlap
  if (x1 <= x2 && right1 > left2) xhit = true;
  if (x1 > x2 && left1 < right2) xhit = true;

  if (y1 <= y2 && btm1 > top2) yhit = true;
  if (y1 > y2 && top1 < btm2) yhit = true;

  if (xhit && yhit) return true;
  return false;
}