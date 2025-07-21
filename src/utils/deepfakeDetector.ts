import * as tf from '@tensorflow/tfjs';
import { ModelLoader } from './modelLoader';
import { ImageProcessor } from './imageProcessor';

export interface DetectionAnalysis {
  faceDetection: number;
  temporalConsistency: number;
  artifactDetection: number;
  imageQuality: number;
  neuralNetworkConfidence: number;
}

export interface DeepfakeDetectionResult {
  isDeepfake: boolean;
  confidence: number;
  threatLevel: 'low' | 'medium' | 'high' | 'critical';
  analysis: DetectionAnalysis;
  processingTime: number;
}

export class DeepfakeDetector {
  private modelLoader: ModelLoader;

  constructor() {
    this.modelLoader = ModelLoader.getInstance();
  }

  async initialize(): Promise<void> {
    await this.modelLoader.initializeModels();
  }

  async detectDeepfake(file: File): Promise<DeepfakeDetectionResult> {
    const startTime = performance.now();
    
    if (!this.modelLoader.isModelsReady()) {
      throw new Error('Models not initialized');
    }

    const deepfakeModel = this.modelLoader.getDeepfakeModel()!;
    const faceModel = this.modelLoader.getFaceDetectionModel()!;

    let analysis: DetectionAnalysis;
    let confidence: number;

    if (file.type.startsWith('image/')) {
      const result = await this.analyzeImage(file, deepfakeModel, faceModel);
      analysis = result.analysis;
      confidence = result.confidence;
    } else if (file.type.startsWith('video/')) {
      const result = await this.analyzeVideo(file, deepfakeModel, faceModel);
      analysis = result.analysis;
      confidence = result.confidence;
    } else {
      throw new Error('Unsupported file type');
    }

    const processingTime = performance.now() - startTime;
    const isDeepfake = confidence > 0.5;
    const threatLevel = this.calculateThreatLevel(confidence, analysis);

    return {
      isDeepfake,
      confidence: confidence * 100,
      threatLevel,
      analysis,
      processingTime
    };
  }

  private async analyzeImage(
    file: File, 
    deepfakeModel: tf.LayersModel, 
    faceModel: tf.LayersModel
  ): Promise<{ confidence: number; analysis: DetectionAnalysis }> {
    const img = await this.loadImage(file);
    const enhancedImg = await ImageProcessor.enhanceImage(img);
    
    // Preprocess image for neural network
    const tensor = await ImageProcessor.preprocessImage(enhancedImg);
    
    // Run deepfake detection
    const prediction = deepfakeModel.predict(tensor) as tf.Tensor;
    const confidence = (await prediction.data())[0];
    
    // Face detection analysis
    const faceCoords = await ImageProcessor.detectFaces(enhancedImg, faceModel);
    const faceDetectionScore = this.calculateFaceDetectionScore(faceCoords);
    
    // Artifact detection using edge analysis
    const artifactScore = await this.detectArtifacts(enhancedImg);
    
    // Image quality assessment
    const imageQuality = ImageProcessor.calculateImageQuality(enhancedImg);
    
    // Clean up tensors
    tensor.dispose();
    prediction.dispose();
    
    const analysis: DetectionAnalysis = {
      faceDetection: faceDetectionScore,
      temporalConsistency: 100, // N/A for images
      artifactDetection: artifactScore,
      imageQuality,
      neuralNetworkConfidence: confidence * 100
    };

    return { confidence, analysis };
  }

  private async analyzeVideo(
    file: File, 
    deepfakeModel: tf.LayersModel, 
    faceModel: tf.LayersModel
  ): Promise<{ confidence: number; analysis: DetectionAnalysis }> {
    const frames = await ImageProcessor.extractFramesFromVideo(file, 8);
    const frameAnalyses: DetectionAnalysis[] = [];
    let totalConfidence = 0;

    for (const frame of frames) {
      const enhancedFrame = await ImageProcessor.enhanceImage(frame);
      const tensor = await ImageProcessor.preprocessImage(enhancedFrame);
      
      const prediction = deepfakeModel.predict(tensor) as tf.Tensor;
      const frameConfidence = (await prediction.data())[0];
      totalConfidence += frameConfidence;
      
      const faceCoords = await ImageProcessor.detectFaces(enhancedFrame, faceModel);
      const faceDetectionScore = this.calculateFaceDetectionScore(faceCoords);
      const artifactScore = await this.detectArtifacts(enhancedFrame);
      const imageQuality = ImageProcessor.calculateImageQuality(enhancedFrame);
      
      frameAnalyses.push({
        faceDetection: faceDetectionScore,
        temporalConsistency: 0, // Will be calculated separately
        artifactDetection: artifactScore,
        imageQuality,
        neuralNetworkConfidence: frameConfidence * 100
      });
      
      tensor.dispose();
      prediction.dispose();
    }

    const avgConfidence = totalConfidence / frames.length;
    const temporalConsistency = this.calculateTemporalConsistency(frameAnalyses);
    
    const analysis: DetectionAnalysis = {
      faceDetection: frameAnalyses.reduce((sum, a) => sum + a.faceDetection, 0) / frameAnalyses.length,
      temporalConsistency,
      artifactDetection: frameAnalyses.reduce((sum, a) => sum + a.artifactDetection, 0) / frameAnalyses.length,
      imageQuality: frameAnalyses.reduce((sum, a) => sum + a.imageQuality, 0) / frameAnalyses.length,
      neuralNetworkConfidence: avgConfidence * 100
    };

    return { confidence: avgConfidence, analysis };
  }

  private async detectArtifacts(imageElement: HTMLImageElement): Promise<number> {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return 0;
    
    canvas.width = imageElement.width;
    canvas.height = imageElement.height;
    ctx.drawImage(imageElement, 0, 0);
    
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // Edge detection using Sobel operator
    let edgeStrength = 0;
    const width = canvas.width;
    const height = canvas.height;
    
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = (y * width + x) * 4;
        
        // Get surrounding pixels
        const tl = data[((y-1) * width + (x-1)) * 4];
        const tm = data[((y-1) * width + x) * 4];
        const tr = data[((y-1) * width + (x+1)) * 4];
        const ml = data[(y * width + (x-1)) * 4];
        const mr = data[(y * width + (x+1)) * 4];
        const bl = data[((y+1) * width + (x-1)) * 4];
        const bm = data[((y+1) * width + x) * 4];
        const br = data[((y+1) * width + (x+1)) * 4];
        
        // Sobel X and Y
        const sobelX = (tr + 2*mr + br) - (tl + 2*ml + bl);
        const sobelY = (bl + 2*bm + br) - (tl + 2*tm + tr);
        
        edgeStrength += Math.sqrt(sobelX*sobelX + sobelY*sobelY);
      }
    }
    
    // Normalize and return artifact score (higher = more artifacts)
    const normalizedEdgeStrength = edgeStrength / (width * height);
    return Math.min(100, normalizedEdgeStrength / 10);
  }

  private calculateFaceDetectionScore(faceCoords: number[]): number {
    // Simple face detection confidence based on bounding box validity
    if (faceCoords.length < 4) return 0;
    
    const [x, y, width, height] = faceCoords;
    
    // Check if coordinates are reasonable
    if (x >= 0 && y >= 0 && width > 0 && height > 0 && 
        x + width <= 1 && y + height <= 1) {
      return Math.min(100, (width * height) * 1000); // Larger faces = higher confidence
    }
    
    return 0;
  }

  private calculateTemporalConsistency(frameAnalyses: DetectionAnalysis[]): number {
    if (frameAnalyses.length < 2) return 100;
    
    let consistencyScore = 0;
    
    for (let i = 1; i < frameAnalyses.length; i++) {
      const prev = frameAnalyses[i - 1];
      const curr = frameAnalyses[i];
      
      // Calculate differences between consecutive frames
      const faceDiff = Math.abs(prev.faceDetection - curr.faceDetection);
      const artifactDiff = Math.abs(prev.artifactDetection - curr.artifactDetection);
      const qualityDiff = Math.abs(prev.imageQuality - curr.imageQuality);
      
      // Lower differences = higher consistency
      const frameConsistency = 100 - ((faceDiff + artifactDiff + qualityDiff) / 3);
      consistencyScore += frameConsistency;
    }
    
    return consistencyScore / (frameAnalyses.length - 1);
  }

  private calculateThreatLevel(confidence: number, analysis: DetectionAnalysis): 'low' | 'medium' | 'high' | 'critical' {
    const score = confidence * 100;
    
    if (score > 90 && analysis.artifactDetection > 70) return 'critical';
    if (score > 75) return 'high';
    if (score > 50) return 'medium';
    return 'low';
  }

  private loadImage(file: File): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }
}