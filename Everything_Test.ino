void setup() {
  Serial.begin(115200);

  pinMode(17, INPUT_PULLDOWN);
  pinMode(18, INPUT_PULLDOWN);
  pinMode(21, INPUT_PULLDOWN);
  pinMode(10, INPUT_PULLDOWN);
}

void loop() {
  Serial.print("P:");
  Serial.print(digitalRead(17));
  Serial.print(" I:");
  Serial.print(digitalRead(18));
  Serial.print(" M:");
  Serial.print(digitalRead(21));
  Serial.print(" UM:");
  Serial.println(digitalRead(10));

  delay(200);
}