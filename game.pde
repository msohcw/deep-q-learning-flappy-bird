enum PhysicsModel{
	MOVE, JUMP, HOVER
}

enum Action{
	UP, DOWN
}

class FlappyBird {
	PhysicsModel world;
	ArrayList<Obstacle> obstacles = new ArrayList<Obstacle>();

	int obstacleGap = 50;
  	int obstacleWidth = 10;
  	int difficulty = 10;

	FlappyBird(PhysicsModel w){
		world = w;
		for(int i = 0; i < stageWidth % (obstacleGap + obstacleWidth); i++){
			addObstacleAt(obstacleGap + (obstacleGap + obstacleWidth) * i);	
		}
	}

	public void nextFrame(){
		// player
		velocity.y += acceleration.y;
		position.y += velocity.y;
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
		PVector gapCenter = new PVector(x, position.y);
		int gapHeight = random(difficulty);

		PVector topLeft = gapCenter.copy();
		topLeft.y += gapHeight/2;

		Obstacle o = new Obstacle(topLeft, gapHeight, obstacleWidth);
		obstacles.add(o);
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

public class Obstacle{
	PVector topLeft;
	float gapHeight;
	float width;

	int speed = 2;

	Obstacle(PVector tl, float h, float w){
		topLeft = tl;
		gapHeight = h;
		width = w;
	}

	public float getX(){
		return topLeft.x;
	}

	public float getGapTop(){
		return topLeft.y + gapHeight;
	}

	public float getGapBottom(){
		return topLeft.y;
	}

	public void advance(int x){
		topLeft.x -= speed;
	}

	public boolean passed(){
		return ((tl.x - w) < 0);
	}
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