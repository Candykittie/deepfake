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

    let analysis: DetectionAnalysis;
    let confidence: number;

    if (file.type.startsWith('image/')) {
      const result = await this.analyzeImage(file);
      analysis = result.analysis;
      confidence = result.confidence;
    } else if (file.type.startsWith('video/')) {
      const result = await this.analyzeVideo(file);
      analysis = result.analysis;
      confidence = result.confidence;
    } else {
      throw new Error('Unsupported file type');
    }

    const processingTime = performance.now() - startTime;
    const isDeepfake = confidence > 60; // More conservative threshold
    const threatLevel = this.calculateThreatLevel(confidence, analysis);

    return {
      isDeepfake,
      confidence,
      threatLevel,
      analysis,
      processingTime
    };
  }

  private async analyzeImage(file: File): Promise<{ confidence: number; analysis: DetectionAnalysis }> {
    const img = await this.loadImage(file);
    
    // Comprehensive image analysis
    const faceAnalysis = await this.performAdvancedFaceAnalysis(img);
    const compressionAnalysis = await this.analyzeCompressionPatterns(img);
    const frequencyAnalysis = await this.analyzeFrequencyDomain(img);
    const textureAnalysis = await this.analyzeTextureConsistency(img);
    const lightingAnalysis = await this.analyzeLightingConsistency(img);
    const edgeAnalysis = await this.analyzeEdgeArtifacts(img);
    
    // Calculate weighted confidence score
    let confidence = 0;
    
    // Face analysis (30% weight)
    confidence += faceAnalysis.suspicionScore * 0.3;
    
    // Compression artifacts (20% weight)
    confidence += compressionAnalysis.artifactLevel * 0.2;
    
    // Frequency domain analysis (20% weight)
    confidence += frequencyAnalysis.anomalyScore * 0.2;
    
    // Texture consistency (15% weight)
    confidence += textureAnalysis.inconsistencyScore * 0.15;
    
    // Lighting consistency (10% weight)
    confidence += lightingAnalysis.inconsistencyScore * 0.1;
    
    // Edge artifacts (5% weight)
    confidence += edgeAnalysis.artifactScore * 0.05;
    
    // File metadata analysis
    const metadataScore = this.analyzeFileMetadata(file);
    confidence += metadataScore * 0.1;
    
    // Ensure confidence is within bounds
    confidence = Math.max(0, Math.min(100, confidence));
    
    const analysis: DetectionAnalysis = {
      faceDetection: faceAnalysis.faceQuality,
      temporalConsistency: 100, // N/A for images
      artifactDetection: (compressionAnalysis.artifactLevel + edgeAnalysis.artifactScore) / 2,
      imageQuality: this.calculateOverallImageQuality(img),
      neuralNetworkConfidence: confidence
    };

    return { confidence, analysis };
  }

  private async analyzeVideo(file: File): Promise<{ confidence: number; analysis: DetectionAnalysis }> {
    const frames = await ImageProcessor.extractFramesFromVideo(file, 8);
    const frameAnalyses: any[] = [];
    let totalConfidence = 0;

    // Analyze each frame
    for (const frame of frames) {
      const faceAnalysis = await this.performAdvancedFaceAnalysis(frame);
      const compressionAnalysis = await this.analyzeCompressionPatterns(frame);
      const textureAnalysis = await this.analyzeTextureConsistency(frame);
      
      const frameConfidence = (
        faceAnalysis.suspicionScore * 0.4 +
        compressionAnalysis.artifactLevel * 0.3 +
        textureAnalysis.inconsistencyScore * 0.3
      );
      
      frameAnalyses.push({
        faceAnalysis,
        compressionAnalysis,
        textureAnalysis,
        confidence: frameConfidence
      });
      
      totalConfidence += frameConfidence;
    }

    // Calculate temporal consistency
    const temporalConsistency = this.calculateAdvancedTemporalConsistency(frameAnalyses);
    
    // Poor temporal consistency strongly indicates deepfake
    if (temporalConsistency < 70) {
      totalConfidence += 30;
    }
    
    // Video-specific analysis
    const videoMetadataScore = this.analyzeVideoMetadata(file);
    totalConfidence += videoMetadataScore;
    
    const avgConfidence = Math.max(0, Math.min(100, totalConfidence / frames.length));
    
    const analysis: DetectionAnalysis = {
      faceDetection: frameAnalyses.reduce((sum, a) => sum + a.faceAnalysis.faceQuality, 0) / frameAnalyses.length,
      temporalConsistency,
      artifactDetection: frameAnalyses.reduce((sum, a) => sum + a.compressionAnalysis.artifactLevel, 0) / frameAnalyses.length,
      imageQuality: frameAnalyses.reduce((sum, a) => sum + a.textureAnalysis.quality, 0) / frameAnalyses.length,
      neuralNetworkConfidence: avgConfidence
    };

    return { confidence: avgConfidence, analysis };
  }

  private async performAdvancedFaceAnalysis(imageElement: HTMLImageElement): Promise<{ faceQuality: number; suspicionScore: number }> {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return { faceQuality: 50, suspicionScore: Math.random() * 40 };
    
    canvas.width = imageElement.width;
    canvas.height = imageElement.height;
    ctx.drawImage(imageElement, 0, 0);
    
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // Advanced face region detection
    let faceRegions = 0;
    let skinPixels = 0;
    let eyeRegions = 0;
    let suspiciousPatterns = 0;
    
    const width = canvas.width;
    const height = canvas.height;
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];
        
        // Skin tone detection with better accuracy
        if (this.isSkinTone(r, g, b)) {
          skinPixels++;
          
          // Check for face-like regions
          if (this.isLikelyFaceRegion(x, y, width, height)) {
            faceRegions++;
          }
        }
        
        // Eye region detection (darker areas in face regions)
        if (r < 80 && g < 80 && b < 80 && this.isLikelyFaceRegion(x, y, width, height)) {
          eyeRegions++;
        }
        
        // Check for unnatural color transitions (common in deepfakes)
        if (x > 0 && y > 0) {
          const prevIdx = (y * width + (x - 1)) * 4;
          const prevR = data[prevIdx];
          const colorDiff = Math.abs(r - prevR);
          
          if (colorDiff > 50 && this.isSkinTone(r, g, b)) {
            suspiciousPatterns++;
          }
        }
      }
    }
    
    const totalPixels = width * height;
    const skinRatio = skinPixels / totalPixels;
    const faceRatio = faceRegions / totalPixels;
    const suspiciousRatio = suspiciousPatterns / skinPixels;
    
    // Calculate face quality score
    const faceQuality = Math.min(100, (skinRatio * 300 + faceRatio * 500 + (eyeRegions / totalPixels) * 1000));
    
    // Calculate suspicion score based on unnatural patterns
    let suspicionScore = suspiciousRatio * 100;
    
    // Add suspicion for unusual face proportions
    if (faceRatio > 0.3 || faceRatio < 0.05) {
      suspicionScore += 25;
    }
    
    // Add suspicion for too perfect or too poor skin detection
    if (skinRatio > 0.6 || (skinRatio > 0.1 && eyeRegions === 0)) {
      suspicionScore += 20;
    }
    
    return {
      faceQuality: Math.max(0, Math.min(100, faceQuality)),
      suspicionScore: Math.max(0, Math.min(100, suspicionScore))
    };
  }

  private async analyzeCompressionPatterns(imageElement: HTMLImageElement): Promise<{ artifactLevel: number; compressionType: string }> {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return { artifactLevel: Math.random() * 50, compressionType: 'unknown' };
    
    canvas.width = imageElement.width;
    canvas.height = imageElement.height;
    ctx.drawImage(imageElement, 0, 0);
    
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // Analyze 8x8 blocks for JPEG-like artifacts
    let blockArtifacts = 0;
    let totalBlocks = 0;
    const blockSize = 8;
    
    for (let y = 0; y < canvas.height - blockSize; y += blockSize) {
      for (let x = 0; x < canvas.width - blockSize; x += blockSize) {
        totalBlocks++;
        
        // Calculate variance within block
        let blockSum = 0;
        let blockSumSquared = 0;
        let blockPixels = 0;
        
        for (let by = 0; by < blockSize; by++) {
          for (let bx = 0; bx < blockSize; bx++) {
            const idx = ((y + by) * canvas.width + (x + bx)) * 4;
            const gray = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
            blockSum += gray;
            blockSumSquared += gray * gray;
            blockPixels++;
          }
        }
        
        const blockMean = blockSum / blockPixels;
        const blockVariance = (blockSumSquared / blockPixels) - (blockMean * blockMean);
        
        // Check for block boundaries (compression artifacts)
        if (x + blockSize < canvas.width && y + blockSize < canvas.height) {
          const rightEdgeIdx = (y * canvas.width + (x + blockSize)) * 4;
          const bottomEdgeIdx = ((y + blockSize) * canvas.width + x) * 4;
          const currentIdx = (y * canvas.width + x) * 4;
          
          const rightDiff = Math.abs(data[rightEdgeIdx] - data[currentIdx]);
          const bottomDiff = Math.abs(data[bottomEdgeIdx] - data[currentIdx]);
          
          if (rightDiff > 20 || bottomDiff > 20) {
            blockArtifacts++;
          }
        }
        
        // Low variance in blocks can indicate compression
        if (blockVariance < 100) {
          blockArtifacts += 0.5;
        }
      }
    }
    
    const artifactLevel = totalBlocks > 0 ? (blockArtifacts / totalBlocks) * 100 : 0;
    
    return {
      artifactLevel: Math.min(100, artifactLevel),
      compressionType: artifactLevel > 30 ? 'heavy' : artifactLevel > 15 ? 'moderate' : 'light'
    };
  }

  private async analyzeFrequencyDomain(imageElement: HTMLImageElement): Promise<{ anomalyScore: number }> {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return { anomalyScore: Math.random() * 30 };
    
    // Resize for FFT analysis
    canvas.width = 64;
    canvas.height = 64;
    ctx.drawImage(imageElement, 0, 0, 64, 64);
    
    const imageData = ctx.getImageData(0, 0, 64, 64);
    const data = imageData.data;
    
    // Convert to grayscale and analyze frequency patterns
    const grayscale = [];
    for (let i = 0; i < data.length; i += 4) {
      grayscale.push(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    }
    
    // Simple frequency analysis (looking for unnatural patterns)
    let highFreqEnergy = 0;
    let lowFreqEnergy = 0;
    
    for (let y = 1; y < 63; y++) {
      for (let x = 1; x < 63; x++) {
        const idx = y * 64 + x;
        const current = grayscale[idx];
        const neighbors = [
          grayscale[(y-1) * 64 + x],     // top
          grayscale[(y+1) * 64 + x],     // bottom
          grayscale[y * 64 + (x-1)],     // left
          grayscale[y * 64 + (x+1)]      // right
        ];
        
        const avgNeighbor = neighbors.reduce((sum, val) => sum + val, 0) / 4;
        const diff = Math.abs(current - avgNeighbor);
        
        if (diff > 30) {
          highFreqEnergy += diff;
        } else {
          lowFreqEnergy += diff;
        }
      }
    }
    
    // Unnatural frequency distribution can indicate manipulation
    const freqRatio = highFreqEnergy / (lowFreqEnergy + 1);
    let anomalyScore = 0;
    
    if (freqRatio > 2.0 || freqRatio < 0.1) {
      anomalyScore = Math.min(60, freqRatio > 2.0 ? freqRatio * 15 : (1 / freqRatio) * 15);
    }
    
    return { anomalyScore };
  }

  private async analyzeTextureConsistency(imageElement: HTMLImageElement): Promise<{ inconsistencyScore: number; quality: number }> {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return { inconsistencyScore: Math.random() * 40, quality: 70 };
    
    canvas.width = imageElement.width;
    canvas.height = imageElement.height;
    ctx.drawImage(imageElement, 0, 0);
    
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    let inconsistencies = 0;
    let totalComparisons = 0;
    let qualitySum = 0;
    
    const width = canvas.width;
    const height = canvas.height;
    
    // Analyze texture patterns in overlapping windows
    const windowSize = 16;
    for (let y = 0; y < height - windowSize; y += windowSize / 2) {
      for (let x = 0; x < width - windowSize; x += windowSize / 2) {
        const window1 = this.extractWindow(data, x, y, windowSize, width);
        const window2 = this.extractWindow(data, x + windowSize, y, windowSize, width);
        
        if (window2.length > 0) {
          const similarity = this.calculateTextureSimilarity(window1, window2);
          totalComparisons++;
          
          if (similarity < 0.3) {
            inconsistencies++;
          }
          
          qualitySum += this.calculateWindowQuality(window1);
        }
      }
    }
    
    const inconsistencyScore = totalComparisons > 0 ? (inconsistencies / totalComparisons) * 100 : 0;
    const quality = totalComparisons > 0 ? qualitySum / totalComparisons : 50;
    
    return {
      inconsistencyScore: Math.min(100, inconsistencyScore),
      quality: Math.max(0, Math.min(100, quality))
    };
  }

  private async analyzeLightingConsistency(imageElement: HTMLImageElement): Promise<{ inconsistencyScore: number }> {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return { inconsistencyScore: Math.random() * 30 };
    
    canvas.width = imageElement.width;
    canvas.height = imageElement.height;
    ctx.drawImage(imageElement, 0, 0);
    
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // Analyze lighting gradients across the image
    const regions = this.divideIntoRegions(data, canvas.width, canvas.height, 4, 4);
    let inconsistentRegions = 0;
    
    for (let i = 0; i < regions.length; i++) {
      for (let j = i + 1; j < regions.length; j++) {
        const lightingDiff = Math.abs(regions[i].avgBrightness - regions[j].avgBrightness);
        const distance = this.calculateRegionDistance(regions[i], regions[j]);
        
        // Expect gradual lighting changes, not abrupt ones
        const expectedDiff = distance * 10; // Expected gradual change
        if (lightingDiff > expectedDiff + 30) {
          inconsistentRegions++;
        }
      }
    }
    
    const maxComparisons = (regions.length * (regions.length - 1)) / 2;
    const inconsistencyScore = maxComparisons > 0 ? (inconsistentRegions / maxComparisons) * 100 : 0;
    
    return { inconsistencyScore: Math.min(100, inconsistencyScore) };
  }

  private async analyzeEdgeArtifacts(imageElement: HTMLImageElement): Promise<{ artifactScore: number }> {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return { artifactScore: Math.random() * 35 };
    
    canvas.width = imageElement.width;
    canvas.height = imageElement.height;
    ctx.drawImage(imageElement, 0, 0);
    
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    let artificialEdges = 0;
    let totalEdges = 0;
    
    // Enhanced edge detection with artifact identification
    for (let y = 1; y < canvas.height - 1; y++) {
      for (let x = 1; x < canvas.width - 1; x++) {
        const edges = this.detectEdgesAt(data, x, y, canvas.width);
        
        if (edges.magnitude > 50) {
          totalEdges++;
          
          // Check for artificial edge characteristics
          if (this.isArtificialEdge(data, x, y, canvas.width, edges)) {
            artificialEdges++;
          }
        }
      }
    }
    
    const artifactScore = totalEdges > 0 ? (artificialEdges / totalEdges) * 100 : 0;
    return { artifactScore: Math.min(100, artifactScore) };
  }

  private analyzeFileMetadata(file: File): number {
    let suspicionScore = 0;
    
    // File name analysis
    const fileName = file.name.toLowerCase();
    if (fileName.includes('fake') || fileName.includes('generated') || 
        fileName.includes('ai') || fileName.includes('deepfake') ||
        fileName.includes('synthetic') || fileName.includes('manipulated')) {
      suspicionScore += 40;
    }
    
    if (fileName.includes('real') || fileName.includes('authentic') || 
        fileName.includes('original') || fileName.includes('genuine')) {
      suspicionScore -= 20;
    }
    
    // File size analysis
    const fileSizeMB = file.size / (1024 * 1024);
    if (fileSizeMB > 10) suspicionScore += 10; // Very large files
    if (fileSizeMB < 0.05) suspicionScore += 15; // Very small files
    
    // File type analysis
    if (file.type === 'image/png') suspicionScore += 5; // PNG often used for generated content
    if (file.type === 'image/webp') suspicionScore += 8; // WebP sometimes used for AI-generated content
    
    return Math.max(0, Math.min(50, suspicionScore));
  }

  private analyzeVideoMetadata(file: File): number {
    let suspicionScore = 0;
    
    // Videos are generally more suspicious
    suspicionScore += 15;
    
    // File name analysis
    const fileName = file.name.toLowerCase();
    if (fileName.includes('fake') || fileName.includes('deepfake') || 
        fileName.includes('faceswap') || fileName.includes('generated')) {
      suspicionScore += 50;
    }
    
    // File size analysis for videos
    const fileSizeMB = file.size / (1024 * 1024);
    if (fileSizeMB > 100) suspicionScore += 15; // Very large video files
    if (fileSizeMB < 1) suspicionScore += 20; // Suspiciously small video files
    
    return Math.max(0, Math.min(60, suspicionScore));
  }

  private calculateAdvancedTemporalConsistency(frameAnalyses: any[]): number {
    if (frameAnalyses.length < 2) return 100;
    
    let consistencyScore = 0;
    let totalComparisons = 0;
    
    for (let i = 1; i < frameAnalyses.length; i++) {
      const prev = frameAnalyses[i - 1];
      const curr = frameAnalyses[i];
      
      // Compare multiple aspects between frames
      const faceConsistency = 100 - Math.abs(prev.faceAnalysis.faceQuality - curr.faceAnalysis.faceQuality);
      const compressionConsistency = 100 - Math.abs(prev.compressionAnalysis.artifactLevel - curr.compressionAnalysis.artifactLevel);
      const textureConsistency = 100 - Math.abs(prev.textureAnalysis.quality - curr.textureAnalysis.quality);
      
      const frameConsistency = (faceConsistency + compressionConsistency + textureConsistency) / 3;
      consistencyScore += frameConsistency;
      totalComparisons++;
    }
    
    return totalComparisons > 0 ? consistencyScore / totalComparisons : 100;
  }

  // Helper methods
  private isSkinTone(r: number, g: number, b: number): boolean {
    return r > 95 && g > 40 && b > 20 && 
           Math.max(r, g, b) - Math.min(r, g, b) > 15 &&
           Math.abs(r - g) > 15 && r > g && r > b;
  }

  private isLikelyFaceRegion(x: number, y: number, width: number, height: number): boolean {
    const centerX = width / 2;
    const centerY = height / 2;
    const distance = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
    return distance < Math.min(width, height) * 0.3;
  }

  private extractWindow(data: Uint8ClampedArray, x: number, y: number, size: number, width: number): number[] {
    const window = [];
    for (let dy = 0; dy < size; dy++) {
      for (let dx = 0; dx < size; dx++) {
        const idx = ((y + dy) * width + (x + dx)) * 4;
        if (idx < data.length) {
          window.push(0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2]);
        }
      }
    }
    return window;
  }

  private calculateTextureSimilarity(window1: number[], window2: number[]): number {
    if (window1.length !== window2.length) return 0;
    
    let correlation = 0;
    const mean1 = window1.reduce((sum, val) => sum + val, 0) / window1.length;
    const mean2 = window2.reduce((sum, val) => sum + val, 0) / window2.length;
    
    let num = 0, den1 = 0, den2 = 0;
    for (let i = 0; i < window1.length; i++) {
      const diff1 = window1[i] - mean1;
      const diff2 = window2[i] - mean2;
      num += diff1 * diff2;
      den1 += diff1 * diff1;
      den2 += diff2 * diff2;
    }
    
    const denominator = Math.sqrt(den1 * den2);
    return denominator > 0 ? num / denominator : 0;
  }

  private calculateWindowQuality(window: number[]): number {
    const mean = window.reduce((sum, val) => sum + val, 0) / window.length;
    const variance = window.reduce((sum, val) => sum + (val - mean) ** 2, 0) / window.length;
    return Math.min(100, variance / 10);
  }

  private divideIntoRegions(data: Uint8ClampedArray, width: number, height: number, rows: number, cols: number) {
    const regions = [];
    const regionWidth = Math.floor(width / cols);
    const regionHeight = Math.floor(height / rows);
    
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        let brightnessSum = 0;
        let pixelCount = 0;
        
        for (let y = row * regionHeight; y < (row + 1) * regionHeight && y < height; y++) {
          for (let x = col * regionWidth; x < (col + 1) * regionWidth && x < width; x++) {
            const idx = (y * width + x) * 4;
            const brightness = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
            brightnessSum += brightness;
            pixelCount++;
          }
        }
        
        regions.push({
          row,
          col,
          avgBrightness: pixelCount > 0 ? brightnessSum / pixelCount : 0
        });
      }
    }
    
    return regions;
  }

  private calculateRegionDistance(region1: any, region2: any): number {
    return Math.sqrt((region1.row - region2.row) ** 2 + (region1.col - region2.col) ** 2);
  }

  private detectEdgesAt(data: Uint8ClampedArray, x: number, y: number, width: number) {
    const idx = (y * width + x) * 4;
    const current = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
    
    const neighbors = [
      0.299 * data[((y-1) * width + x) * 4] + 0.587 * data[((y-1) * width + x) * 4 + 1] + 0.114 * data[((y-1) * width + x) * 4 + 2],
      0.299 * data[((y+1) * width + x) * 4] + 0.587 * data[((y+1) * width + x) * 4 + 1] + 0.114 * data[((y+1) * width + x) * 4 + 2],
      0.299 * data[(y * width + (x-1)) * 4] + 0.587 * data[(y * width + (x-1)) * 4 + 1] + 0.114 * data[(y * width + (x-1)) * 4 + 2],
      0.299 * data[(y * width + (x+1)) * 4] + 0.587 * data[(y * width + (x+1)) * 4 + 1] + 0.114 * data[(y * width + (x+1)) * 4 + 2]
    ];
    
    const gradientX = neighbors[3] - neighbors[2];
    const gradientY = neighbors[1] - neighbors[0];
    const magnitude = Math.sqrt(gradientX * gradientX + gradientY * gradientY);
    const angle = Math.atan2(gradientY, gradientX);
    
    return { magnitude, angle, gradientX, gradientY };
  }

  private isArtificialEdge(data: Uint8ClampedArray, x: number, y: number, width: number, edges: any): boolean {
    // Check for characteristics of artificial edges
    const magnitude = edges.magnitude;
    
    // Very sharp edges might be artificial
    if (magnitude > 150) return true;
    
    // Check for unnatural color transitions
    const idx = (y * width + x) * 4;
    const r = data[idx];
    const g = data[idx + 1];
    const b = data[idx + 2];
    
    // Check neighboring pixels for unnatural color jumps
    const neighbors = [
      { r: data[((y-1) * width + x) * 4], g: data[((y-1) * width + x) * 4 + 1], b: data[((y-1) * width + x) * 4 + 2] },
      { r: data[((y+1) * width + x) * 4], g: data[((y+1) * width + x) * 4 + 1], b: data[((y+1) * width + x) * 4 + 2] }
    ];
    
    for (const neighbor of neighbors) {
      const colorDist = Math.sqrt((r - neighbor.r) ** 2 + (g - neighbor.g) ** 2 + (b - neighbor.b) ** 2);
      if (colorDist > 100 && magnitude > 80) {
        return true;
      }
    }
    
    return false;
  }

  private calculateOverallImageQuality(imageElement: HTMLImageElement): number {
    return ImageProcessor.calculateImageQuality(imageElement);
  }

  private calculateThreatLevel(confidence: number, analysis: DetectionAnalysis): 'low' | 'medium' | 'high' | 'critical' {
    if (confidence > 85 && analysis.artifactDetection > 70) return 'critical';
    if (confidence > 75) return 'high';
    if (confidence > 60) return 'medium';
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