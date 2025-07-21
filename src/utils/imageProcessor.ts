import * as tf from '@tensorflow/tfjs';

export class ImageProcessor {
  static async preprocessImage(imageElement: HTMLImageElement): Promise<tf.Tensor4D> {
    // Convert image to tensor and normalize
    const tensor = tf.browser.fromPixels(imageElement)
      .resizeNearestNeighbor([224, 224])
      .toFloat()
      .div(255.0)
      .expandDims(0) as tf.Tensor4D;

    return tensor;
  }

  static async extractFramesFromVideo(videoFile: File, maxFrames: number = 10): Promise<HTMLImageElement[]> {
    return new Promise((resolve, reject) => {
      const video = document.createElement('video');
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      const frames: HTMLImageElement[] = [];

      video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        const duration = video.duration;
        const interval = duration / maxFrames;
        let currentTime = 0;
        let frameCount = 0;

        const extractFrame = () => {
          video.currentTime = currentTime;
        };

        video.onseeked = () => {
          if (ctx) {
            ctx.drawImage(video, 0, 0);
            
            const img = new Image();
            img.onload = () => {
              frames.push(img);
              frameCount++;
              
              if (frameCount < maxFrames) {
                currentTime += interval;
                extractFrame();
              } else {
                resolve(frames);
              }
            };
            img.src = canvas.toDataURL();
          }
        };

        extractFrame();
      };

      video.onerror = reject;
      video.src = URL.createObjectURL(videoFile);
    });
  }

  static async detectFaces(imageElement: HTMLImageElement, model: tf.LayersModel): Promise<number[]> {
    const tensor = await this.preprocessImage(imageElement);
    const prediction = model.predict(tensor) as tf.Tensor;
    const result = await prediction.data();
    
    tensor.dispose();
    prediction.dispose();
    
    return Array.from(result);
  }

  static async enhanceImage(imageElement: HTMLImageElement): Promise<HTMLImageElement> {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    if (!ctx) throw new Error('Canvas context not available');
    
    canvas.width = imageElement.width;
    canvas.height = imageElement.height;
    
    // Draw original image
    ctx.drawImage(imageElement, 0, 0);
    
    // Apply enhancement filters
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // Contrast and brightness enhancement
    for (let i = 0; i < data.length; i += 4) {
      // Increase contrast
      data[i] = Math.min(255, Math.max(0, (data[i] - 128) * 1.2 + 128));     // Red
      data[i + 1] = Math.min(255, Math.max(0, (data[i + 1] - 128) * 1.2 + 128)); // Green
      data[i + 2] = Math.min(255, Math.max(0, (data[i + 2] - 128) * 1.2 + 128)); // Blue
    }
    
    ctx.putImageData(imageData, 0, 0);
    
    const enhancedImg = new Image();
    enhancedImg.src = canvas.toDataURL();
    
    return new Promise((resolve) => {
      enhancedImg.onload = () => resolve(enhancedImg);
    });
  }

  static calculateImageQuality(imageElement: HTMLImageElement): number {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return 0;
    
    canvas.width = imageElement.width;
    canvas.height = imageElement.height;
    ctx.drawImage(imageElement, 0, 0);
    
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // Calculate variance (higher variance = better quality)
    let sum = 0;
    let sumSquared = 0;
    
    for (let i = 0; i < data.length; i += 4) {
      const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      sum += gray;
      sumSquared += gray * gray;
    }
    
    const mean = sum / (data.length / 4);
    const variance = (sumSquared / (data.length / 4)) - (mean * mean);
    
    // Normalize to 0-100 scale
    return Math.min(100, variance / 10);
  }
}