//different specific libraries
#include <ros.h>
#include <ros/time.h>

#include <geometry_msgs/Twist.h>

#include "Arduino.h"
#include "AX12A.h"

#define DirectionPin (10u)
#define BaudRate (1000000ul)
#define ID1 (15u)
#define ID2 (6u)

// create the ros node nh. The node will be used to publish to Arduino
ros::NodeHandle nh;

float transVelocity;
float rotVelocity;

float wheelSep = 0.165; // 16.5cm
float wheelRadius = 0.07; // 70mm
double velDiff;
double leftPower;
double rightPower;

double speed1;
double speed2;

void messageCb(const geometry_msgs::Twist &msg)
{
transVelocity = msg.linear.x;
rotVelocity = msg.angular.z;

velDiff = (wheelSep * rotVelocity) / 2.0;
leftPower = (transVelocity + velDiff) / wheelRadius;
rightPower = (transVelocity - velDiff) / wheelRadius;

motorDirection();
}

void motorDirection()

{
if((abs(leftPower) > 0) && (abs(rightPower) > 0))
{
// set the speed
// speed = (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
speed2 = (abs(leftPower) - 0.00) * (1000 - 200) / (6.49 - 0.00) + 200;
speed1 = (abs(rightPower) - 0.00) * (1000 - 200) / (6.49 - 0.00) + 200;

if ((leftPower > 0) && (rightPower > 0))
{
moveForward(speed1, speed2);
}
else if ((leftPower > 0) && (rightPower < 0))
{
moveLeft(speed1, speed2);
}
else if ((leftPower < 0) && (rightPower < 0))
{
moveBackward(speed1, speed2);
}
else if ((leftPower < 0) && (rightPower > 0))
{
moveRight(speed1, speed2);
}
}
else
{
moveStop();
}
}

void moveForward(int speed1, int speed2)
{
ax12a.turn(ID1, RIGHT, speed1); // MAX SPEED: TRY and ERROR ~ 1000
ax12a.turn(ID2, LEFT, speed2);
}

void moveBackward(int speed1, int speed2)
{
ax12a.turn(ID1, LEFT, speed1);
ax12a.turn(ID2, RIGHT, speed2);
}

void moveRight(int speed1, int speed2)

{
ax12a.turn(ID1, LEFT, speed1);
ax12a.turn(ID2, LEFT, speed2);
}

void moveLeft(int speed1, int speed2)
{
ax12a.turn(ID1, RIGHT, speed1);
ax12a.turn(ID2, RIGHT, speed2);
}

void moveStop()
{
ax12a.turn(ID1, LEFT, 0);
ax12a.turn(ID2, LEFT, 0);

// or
// ax12a.turn(ID1, RIGHT, 0);
// ax12a.turn(ID2, RIGHT, 0);
}

ros::Subscriber<geometry_msgs::Twist> sub("/cmd_vel", &messageCb);

void setup()
{
ax12a.begin(BaudRate, DirectionPin, &Serial1); // Using HardwareSerial (Serial1 or Serial2 or Serial3) of ARDUINO MEGA 2560
ax12a.setEndless(ID1, ON);
ax12a.setEndless(ID2, ON);

nh.initNode();
nh.subscribe(sub);
}

void loop()
{
nh.spinOnce();
delay(1);
}
