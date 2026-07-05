// EncoderInterfaceT4
//
//  Read the encoder and translate to a distance and send over USB-Serial and PWM DAC
//  Teensy 4.0 with Teensy Extensions

//  Encoder A - pin 14
//  Encoder B - pin 15
//  Encoder VCC - Vin
//  Encoder ground - GND
//  PWM Analog Out - pin 3
//  Direction pin - pin 21
//
// Original Author:
// Steve Sawtelle
// jET
// Janelia
// HHMI 
//
//Updated by: Brenna Manuel, Sipe Lab, Penn State University, 2026

#define SHOW_MICROS    // if defined, print micros() with each output
#define SHOW_REVERSE   // if this is defined, show abs value of speed, else ignore reverse distance/speed
#define SHOW_ZERO      // if this is defined, show 0 speed if no movement in SPEED_TIMEOUT micros

#define UPDATE_USECS 20000   // don't print faster than this many micros (20ms = 500Hz)
#define SPEED_TIMEOUT 30000   // if we don't move in this many micros assume we are stopped and show 0.0 speed 
#define ZERO_UPDATE_USECS 2000  // send zero speed updates at this interval when stopped

#define VERSION "20240909_fast"
// ===== VERSIONS ======

// 20240909_fast
// - Optimized for closed-loop control with 2ms updates (500Hz)
// - Fast zero speed detection (5ms timeout)
// - Single zero speed notification to prevent serial flooding
// - Minimal latency for real-time control

// 20240909_fixed
// - Fixed zero speed reporting to continuously send 0.0 when stopped
// - Added consistent timestamp reporting for zero speed
// - Fixed output format consistency

// 20240909 sws
// - from EncoderInterfaceT3, changes to use Teensy 4.0
//   pin changes
//   add in spped and dir outputs, speed is PWM analog

// 20230621 sws
// - add ability to compile for cylinder treadmill

// 20230523 sws
// - remove debug output in loop
// - add '?' query for version

// 20220419 sws
// - update lastUsecs during zero speed time so that when the treadmill starts again the speed 
//   is based on the end of last zero time out

// 20221212 sws from suggestions by york at Labmaker
// - strict even intervals causes poor low speed granularity
// - so, return to incremental updates, but limit to no faster than UPDATE_USECS 
// - make define to allow showing zero if no movement in SPEED_TIMEOUT micros
// - add define to allow printing of micros() value

// 20221017 sws
// - add define to print distance and speed at even intervals. 
//     To use allow #define TIMED_UPDATE and UPDATE_RATE sets the rate in updates per second

// 20220720 sws
// - also need to use pin 3 not 0 in LC as 0 doesn't have interrupt capability
// - to keep backward compatibility, set the unused pin as input so they can be tied
//    together on PCB

// 20220711 sws
// - provide alternate compile for Teensy LC, A12 not A14 for DAC

// 20211109 sws
// - was printing absolute run speed, now it shows as negative if reverse

// 20211021 sws
// - add define for offsetting DAC to show reverse speed 

// 20180514 sws
// - DIR pin was set to 12 but hardware has it on 2
// - also had index pin as 2 should be 14

// 20180328 sws
// - remove wait for serial since it will not move past it if no serial port is defined

// 20180319 sws
// - add define to allow showing reverse direction

 
#define encAPin 14 // 12
#define encBPin 15 // 11


// counts per rotation of treadmill encoder wheel
// 200 counts per rev
// 1.03" per rev
// so - 1.03 * 25.4 * pi / 200 /1000 = microns/cnt
#define MM_PER_COUNT 410950  // actually 1/10^6mm per count since we divide by usecs

#define DIST_PER_COUNT ((float)MM_PER_COUNT/1000000.0)   //(float)0.41095
 

static float runSpeed = 0;
volatile uint32_t lastUsecs;
volatile uint32_t thisUsecs;
volatile uint32_t encoderUsecs;
volatile float distance = 0;
volatile int8_t encoderCounts = 0;

#define FW 1
#define BW -1

int dir = FW;

float lastDistance;
float zeroDistance;
volatile boolean movement = false;
boolean lastZeroOutput = false;  // Flag to track if we've sent zero speed (false = not sent, true = sent)

// ------------------------------------------
// interrupt routine for ENCODER_A rising edge (and falling if cylinder)
// ---------------------------------------------
void encoderInt()
{   
   thisUsecs = micros();
  
  int ENCA = digitalReadFast(encAPin);  // always update output 
  int ENCB = digitalReadFast(encBPin); 
  // figure out the direction  
    
  if (ENCA == ENCB )  // if same, then backwards
  {   
    dir = BW;
    #ifdef SHOW_REVERSE 
        distance -= DIST_PER_COUNT;
    #endif
  }  
  else
  {
      dir = FW;
      distance += DIST_PER_COUNT;
  }  
   
  #ifndef SHOW_REVERSE   // if not showing reverse speed force reverse to 0
    if( dir == BW ) runSpeed = 0;      
  #endif  

   movement = true;  
     
}


// ------------------------------------------
// interrupt routine for ENCODER_B rising/falling edge - for 8" cylinder
// ---------------------------------------------
void encoderBInt()
{   
   thisUsecs = micros();
  
  int ENCA = digitalReadFast(encAPin);  // always update output 
  int ENCB = digitalReadFast(encBPin); 
  // figure out the direction  
    
  if (ENCA != ENCB )  // if same, then backwards
  {   
    dir = BW;
    #ifdef SHOW_REVERSE 
        distance -= DIST_PER_COUNT;
    #endif
  }  
  else
  {
      dir = FW;
      distance += DIST_PER_COUNT;
  }  
   
  #ifndef SHOW_REVERSE   // if not showing reverse speed force reverse to 0
    if( dir == BW ) runSpeed = 0;      
  #endif  

   movement = true;  
     
}


void setup()
{
  Serial.begin(192000);
  // while( !Serial);  // if no USB serial connection, this will hang the program
  
  pinMode(encAPin, INPUT_PULLUP); // sets the digital pin as input
  pinMode(encBPin, INPUT_PULLUP); // sets the digital pin as input
  
  lastUsecs = micros();
  thisUsecs = lastUsecs;
  lastDistance = 0;
  zeroDistance = 0;
  lastZeroOutput = false;
  attachInterrupt(encAPin, encoderInt, RISING); // check encoder every A pin rising edge only (like original)
}

boolean moved = false;

 void loop() 
{  

  if( Serial.available() )
  {
     int8_t charin = Serial.read();
     if ( charin == '?' )
     {
       Serial.print("Treadmill V:");
       Serial.println(VERSION);
       #ifdef SHOW_MICROS 
          Serial.print("micros,");
       #endif
       Serial.println("distance (mm),speed (mm/s)");     
     } 
  }
  
  noInterrupts();
  uint32_t newUsecs = thisUsecs;
  float cumDistance = distance;
  moved = movement;
  movement = false;
  interrupts();
  
  uint32_t currentTime = micros();
  
  if( (moved) && ((newUsecs - lastUsecs) > UPDATE_USECS ) ) // are we at least past max update rate
  {
  #ifndef SHOW_REVERSE 
     if( (cumDistance - lastDistance) > 0 )
  #endif     
     { 
       int32_t usecs = newUsecs - lastUsecs;        
       lastUsecs = newUsecs;  
       runSpeed = (1e6 * (cumDistance - lastDistance) ) / (float)usecs;
       
       // Only send update if there's meaningful distance change (threshold to handle floating point precision)
       float distanceChange = cumDistance - lastDistance;
       if(distanceChange > 0.001 || distanceChange < -0.001) // Only send if distance changed by more than 0.001mm
       {
         #ifdef SHOW_MICROS     
           Serial.print(newUsecs);   
           Serial.print(","); 
         #endif     
         Serial.print(cumDistance);
         Serial.print(",");
         Serial.println( runSpeed); 
             
         lastZeroOutput = false;  // Reset zero flag when we have movement
       }
       
       lastDistance = cumDistance;
     }
     
  }
     
  #ifdef SHOW_ZERO
  else 
  {
     if( (currentTime - lastUsecs) > SPEED_TIMEOUT )
     {
       // Send zero speed update only once when we first detect stop
       if( lastZeroOutput == false )
       {
         #ifdef SHOW_MICROS        
           Serial.print(currentTime);  
           Serial.print(",");           
         #endif         
         Serial.print(cumDistance);  // Use current distance, not lastDistance
         Serial.print(",");
         Serial.println("0.0");  
           
         lastZeroOutput = true;  // Mark that we've sent the zero
       }  
     }   
  } 
  #endif 


} // end loop




