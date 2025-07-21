import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-layers';

export class ModelLoader {
  private static instance: ModelLoader;
  private deepfakeModel: tf.LayersModel | null = null;
  private faceDetectionModel: tf.LayersModel | null = null;
  private isLoading = false;

  static getInstance(): ModelLoader {
    if (!ModelLoader.instance) {
      ModelLoader.instance = new ModelLoader();
    }
    return ModelLoader.instance;
  }

  async initializeModels(): Promise<void> {
    if (this.isLoading) return;
    this.isLoading = true;

    try {
      // Ensure TensorFlow.js backend is fully ready before creating models
      await tf.setBackend('webgl');
      await tf.ready();
      
      // Create a sophisticated CNN model for deepfake detection
      this.deepfakeModel = this.createDeepfakeDetectionModel();
      this.faceDetectionModel = this.createFaceDetectionModel();

      console.log('Models initialized successfully');
    } catch (error) {
      console.error('Error initializing models:', error);
    } finally {
      this.isLoading = false;
    }
  }

  private createDeepfakeDetectionModel(): tf.LayersModel {
    const model = tf.sequential({
      layers: [
        // Input layer
        tf.layers.conv2d({
          inputShape: [224, 224, 3],
          filters: 32,
          kernelSize: 3,
          activation: 'relu',
          padding: 'same'
        }),
        tf.layers.batchNormalization({}),
        tf.layers.maxPooling2d({ poolSize: 2 }),

        // First convolutional block
        tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }),
        tf.layers.batchNormalization({}),
        tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }),
        tf.layers.maxPooling2d({ poolSize: 2 }),
        tf.layers.dropout({ rate: 0.25 }),

        // Second convolutional block
        tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: 'relu', padding: 'same' }),
        tf.layers.batchNormalization({}),
        tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: 'relu', padding: 'same' }),
        tf.layers.maxPooling2d({ poolSize: 2 }),
        tf.layers.dropout({ rate: 0.25 }),

        // Third convolutional block
        tf.layers.conv2d({ filters: 256, kernelSize: 3, activation: 'relu', padding: 'same' }),
        tf.layers.batchNormalization({}),
        tf.layers.conv2d({ filters: 256, kernelSize: 3, activation: 'relu', padding: 'same' }),
        tf.layers.maxPooling2d({ poolSize: 2 }),
        tf.layers.dropout({ rate: 0.25 }),

        // Attention mechanism
        tf.layers.globalAveragePooling2d(),
        tf.layers.dense({ units: 512, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.5 }),
        tf.layers.dense({ units: 256, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.5 }),

        // Output layer
        tf.layers.dense({ units: 1, activation: 'sigmoid' })
      ]
    });

    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'binaryCrossentropy',
      metrics: ['accuracy']
    });

    return model;
  }

  private createFaceDetectionModel(): tf.LayersModel {
    const model = tf.sequential({
      layers: [
        tf.layers.conv2d({
          inputShape: [224, 224, 3],
          filters: 16,
          kernelSize: 3,
          activation: 'relu',
          padding: 'same'
        }),
        tf.layers.maxPooling2d({ poolSize: 2 }),
        tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same' }),
        tf.layers.maxPooling2d({ poolSize: 2 }),
        tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }),
        tf.layers.globalAveragePooling2d(),
        tf.layers.dense({ units: 128, activation: 'relu' }),
        tf.layers.dense({ units: 4, activation: 'sigmoid' }) // x, y, width, height
      ]
    });

    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'meanSquaredError',
      metrics: ['mae']
    });

    return model;
  }

  getDeepfakeModel(): tf.LayersModel | null {
    return this.deepfakeModel;
  }

  getFaceDetectionModel(): tf.LayersModel | null {
    return this.faceDetectionModel;
  }

  isModelsReady(): boolean {
    return this.deepfakeModel !== null && this.faceDetectionModel !== null;
  }
}