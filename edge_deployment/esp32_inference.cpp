#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Include the generated TFLite model array (e.g., converted using xxd -i model_quant.tflite > model_data.h)
#include "model_data.h"

// Globals
namespace {
    tflite::ErrorReporter* error_reporter = nullptr;
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;

    // Create an area of memory to use for input, output, and intermediate arrays.
    // Given the Tiny Neural Network (Input -> 8 -> 1), 4KB is generous enough.
    constexpr int kTensorArenaSize = 4 * 1024;
    uint8_t tensor_arena[kTensorArenaSize];
}

void setup() {
    Serial.begin(115200);

    // Set up logging
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    // Map the model into a usable data structure.
    model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(error_reporter,
                             "Model provided is schema version %d not equal "
                             "to supported version %d.",
                             model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // This pulls in all operations, which is fine for our small memory footprint
    static tflite::AllOpsResolver resolver;

    // Build an interpreter to run the model with
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
        return;
    }

    // Obtain pointers to the model's input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    Serial.println("TFLite Model Engine Setup complete.");
}

void loop() {
    // Simulated Preprocessed Sensor Data reading 
    // In a real scenario, this would involve reading from I2C/SPI sensors 
    // and scaling them using the parameters from `StandardScaler`
    
    // Assuming INT8 Quantized model
    // Format: input->data.int8
    
    // Example: Dummy Feature array length
    int input_length = input->dims->data[1]; 
    
    for (int i = 0; i < input_length; i++) {
        // Assigning dummy values (normalized/quantized range -128 to 127)
        input->data.int8[i] = 0; // Represents mean value essentially
    }

    // Run inference
    unsigned long start_time = micros();
    TfLiteStatus invoke_status = interpreter->Invoke();
    unsigned long inference_time = micros() - start_time;

    if (invoke_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
        return;
    }

    // Output is INT8, we must dequantize to get the 0.0 -> 1.0 probability
    // Assuming Sigmoid output [0, 1] mapped to INT8
    int8_t y_quantized = output->data.int8[0];
    
    float y_pred = (y_quantized - output->params.zero_point) * output->params.scale;

    Serial.print("Inference Time (us): ");
    Serial.println(inference_time);
    
    if (y_pred > 0.5) {
        Serial.print("Status: FAILURE DETECTED 🚨 | Confidence: ");
    } else {
        Serial.print("Status: NORMAL ✅ | Confidence: ");
    }
    Serial.println(y_pred * 100);

    delay(2000); // Wait 2 seconds before next inference
}
