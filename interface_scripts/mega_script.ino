#include <CapacitiveSensor.h>

//  *** MAIN INTERFACE SCRIPT ARDUINO MEGA ***
// Sipe Lab, Penn State University, 2026
// Brenna Manuel

const int puffPin = 7;
const int rewardPin = 8;

CapacitiveSensor Spout = CapacitiveSensor(12, 13);

bool puff_active = false;
bool reward_active = false;

unsigned long puff_start_time = 0;
unsigned long reward_start_time = 0;

unsigned long puff_duration = 10;
unsigned long reward_duration = 10;

unsigned long last_sensor_read = 0;
const unsigned long sensor_interval = 20; //was 20

void setup() {
  pinMode(puffPin, OUTPUT);
  pinMode(rewardPin, OUTPUT);
  digitalWrite(puffPin, LOW);
  digitalWrite(rewardPin, LOW);
  Serial.begin(115200);
  Serial.setTimeout(20); //was 10

}

void loop() {
  unsigned long current_time = millis();

  // --- Handle serial input ---
  if (Serial.available() > 0) {
    long input = Serial.parseInt();  // e.g., 12500 (1 = solenoid ID, 2500 = duration)

    if (input >= 10) {  // Require at least 2 digits
      String inputStr = String(input);
      int solenoid_id = inputStr.substring(0, 1).toInt();
      int duration = inputStr.substring(1).toInt();

      if (solenoid_id == 1 && !puff_active) {
        puff_duration = duration;
        digitalWrite(puffPin, HIGH);
        puff_start_time = current_time;
        puff_active = true;
      } else if (solenoid_id == 2 && !reward_active) {
        reward_duration = duration;
        digitalWrite(rewardPin, HIGH);
        reward_start_time = current_time;
        reward_active = true;
      }
    }
  }

  // --- Turn off solenoids after timeout ---
  if (puff_active && (current_time - puff_start_time >= puff_duration)) {
    digitalWrite(puffPin, LOW);
    puff_active = false;
  }

  if (reward_active && (current_time - reward_start_time >= reward_duration)) {
    digitalWrite(rewardPin, LOW);
    reward_active = false;
  }

  // --- Read capacitive sensor ---
  if (current_time - last_sensor_read >= sensor_interval) {
    last_sensor_read = current_time;
    long cap_value = Spout.capacitiveSensor(10);
    Serial.print(current_time);
    Serial.print(",");
    Serial.println(cap_value);
  }
}

