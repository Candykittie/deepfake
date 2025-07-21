import React, { useCallback, useState } from 'react';
import { Upload, File, AlertCircle, Loader2 } from 'lucide-react';
import { DeepfakeDetector } from '../utils/deepfakeDetector';

interface UploadZoneProps {
  onFilesUpload: (files: File[]) => void;
  isProcessing: boolean;
  isModelReady: boolean;
}

const UploadZone: React.FC<UploadZoneProps> = ({ onFilesUpload, isProcessing, isModelReady }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    const validFiles = files.filter(file => 
      file.type.startsWith('image/') || file.type.startsWith('video/')
    );
    
    if (validFiles.length > 0) {
      setUploadedFiles(validFiles);
      onFilesUpload(validFiles);
    }
  }, [onFilesUpload]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      setUploadedFiles(files);
      onFilesUpload(files);
    }
  }, [onFilesUpload]);

  return (
    <div className="space-y-6">
      {/* Upload Zone */}
      <div
        className={`border-2 border-dashed rounded-lg p-12 text-center transition-all duration-300 ${
          isDragOver
            ? 'border-cyan-400 bg-cyan-400/10'
            : 'border-gray-600 hover:border-gray-500'
        } ${isProcessing || !isModelReady ? 'opacity-50 pointer-events-none' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          type="file"
          multiple
          accept="image/*,video/*"
          onChange={handleFileInput}
          className="hidden"
          id="file-upload"
          disabled={isProcessing || !isModelReady}
        />
        
        <div className="space-y-4">
          {isProcessing ? (
            <Loader2 className="mx-auto h-12 w-12 text-cyan-400 animate-spin" />
          ) : !isModelReady ? (
            <AlertCircle className="mx-auto h-12 w-12 text-yellow-400" />
          ) : (
            <Upload className="mx-auto h-12 w-12 text-gray-400" />
          )}
          
          <div>
            <p className="text-xl font-semibold text-white mb-2">
              {isProcessing ? 'AI Processing Files...' : 
               !isModelReady ? 'Loading ML Models...' : 
               'Drop files here or click to upload'}
            </p>
            <p className="text-gray-400">
              {!isModelReady ? 'Neural networks are initializing...' : 
               'Supports JPG, PNG, MP4, AVI files up to 100MB'}
            </p>
          </div>
          
          {!isProcessing && isModelReady && (
            <label
              htmlFor="file-upload"
              className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-white bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 cursor-pointer transition-all duration-200 hover:scale-105"
            >
              <Upload className="h-5 w-5 mr-2" />
              Select Files
            </label>
          )}
        </div>
      </div>

      {/* Security Notice */}
      <div className="bg-yellow-900/20 border border-yellow-600/30 rounded-lg p-4">
        <div className="flex items-start">
          <AlertCircle className="h-5 w-5 text-yellow-400 mt-0.5 mr-3" />
          <div>
            <h3 className="text-sm font-medium text-yellow-400">Security Notice</h3>
            <p className="mt-1 text-sm text-yellow-200">
              All files are processed locally using TensorFlow.js. Advanced CNN models analyze images/videos 
              for deepfake detection without transmitting data to external servers.
            </p>
          </div>
        </div>
      </div>

      {/* File List */}
      {uploadedFiles.length > 0 && (
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Uploaded Files</h3>
          <div className="space-y-3">
            {uploadedFiles.map((file, index) => (
              <div key={index} className="flex items-center space-x-3 p-3 bg-gray-700 rounded-lg">
                <File className="h-5 w-5 text-gray-400" />
                <div className="flex-1">
                  <p className="text-sm font-medium text-white">{file.name}</p>
                  <p className="text-xs text-gray-400">
                    {file.type} • {(file.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
                {isProcessing && (
                  <Loader2 className="h-4 w-4 text-cyan-400 animate-spin" />
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default UploadZone;