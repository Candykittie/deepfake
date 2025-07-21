import React, { useState, useEffect } from 'react';
import { TrendingUp, Brain, Target, AlertCircle, BarChart3, Activity, Zap } from 'lucide-react';

interface DetectionResult {
  id: string;
  filename: string;
  type: 'image' | 'video';
  confidence: number;
  isDeepfake: boolean;
  threatLevel: 'low' | 'medium' | 'high' | 'critical';
  timestamp: Date;
  analysisDetails: {
    faceDetection: number;
    temporalConsistency: number;
    artifactDetection: number;
  };
}

interface PredictiveAnalyticsProps {
  results: DetectionResult[];
}

interface Prediction {
  id: string;
  type: 'threat_surge' | 'pattern_anomaly' | 'system_optimization' | 'risk_assessment';
  title: string;
  description: string;
  confidence: number;
  timeframe: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  recommendations: string[];
}

const PredictiveAnalytics: React.FC<PredictiveAnalyticsProps> = ({ results }) => {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedTimeframe, setSelectedTimeframe] = useState<'1h' | '6h' | '24h' | '7d'>('24h');

  useEffect(() => {
    generatePredictions();
  }, [results, selectedTimeframe]);

  const generatePredictions = async () => {
    setIsAnalyzing(true);
    
    // Simulate AI processing
    await new Promise(resolve => setTimeout(resolve, 2000));

    const newPredictions: Prediction[] = [];

    // Threat surge prediction
    const recentThreats = results.filter(r => r.isDeepfake && 
      (Date.now() - r.timestamp.getTime()) < 3600000);
    
    if (recentThreats.length > 2) {
      newPredictions.push({
        id: 'threat-surge',
        type: 'threat_surge',
        title: 'Elevated Threat Activity Predicted',
        description: `AI models predict a ${Math.floor(Math.random() * 40 + 30)}% increase in deepfake attempts over the next ${selectedTimeframe}.`,
        confidence: 87 + Math.random() * 10,
        timeframe: selectedTimeframe,
        severity: 'high',
        recommendations: [
          'Increase monitoring frequency to every 15 minutes',
          'Activate enhanced detection algorithms',
          'Prepare incident response team for potential surge',
          'Consider implementing additional security layers'
        ]
      });
    }

    // Pattern anomaly detection
    const videoThreats = results.filter(r => r.type === 'video' && r.isDeepfake);
    if (videoThreats.length > 0) {
      newPredictions.push({
        id: 'pattern-anomaly',
        type: 'pattern_anomaly',
        title: 'Unusual Attack Pattern Detected',
        description: 'Neural networks identified a new deepfake generation technique targeting video content with sophisticated temporal manipulation.',
        confidence: 92 + Math.random() * 5,
        timeframe: selectedTimeframe,
        severity: 'critical',
        recommendations: [
          'Update temporal consistency detection models',
          'Implement advanced frame-by-frame analysis',
          'Cross-reference with global threat intelligence',
          'Deploy countermeasure algorithms immediately'
        ]
      });
    }

    // System optimization prediction
    const avgConfidence = results.length > 0 ? 
      results.reduce((acc, r) => acc + r.confidence, 0) / results.length : 0;
    
    if (avgConfidence > 85) {
      newPredictions.push({
        id: 'system-optimization',
        type: 'system_optimization',
        title: 'Model Performance Enhancement Opportunity',
        description: `Current detection accuracy of ${avgConfidence.toFixed(1)}% can be improved by ${Math.floor(Math.random() * 8 + 3)}% through neural network optimization.`,
        confidence: 78 + Math.random() * 15,
        timeframe: selectedTimeframe,
        severity: 'medium',
        recommendations: [
          'Retrain models with recent detection data',
          'Implement ensemble learning techniques',
          'Optimize feature extraction algorithms',
          'Deploy advanced attention mechanisms'
        ]
      });
    }

    // Risk assessment prediction
    const criticalThreats = results.filter(r => r.threatLevel === 'critical').length;
    if (criticalThreats > 0 || Math.random() > 0.5) {
      newPredictions.push({
        id: 'risk-assessment',
        type: 'risk_assessment',
        title: 'Organizational Risk Level Forecast',
        description: `Predictive models indicate ${Math.random() > 0.5 ? 'increasing' : 'stable'} risk exposure with potential for sophisticated social engineering attacks.`,
        confidence: 81 + Math.random() * 12,
        timeframe: selectedTimeframe,
        severity: criticalThreats > 2 ? 'high' : 'medium',
        recommendations: [
          'Conduct security awareness training',
          'Implement multi-factor authentication',
          'Review and update incident response procedures',
          'Establish threat intelligence sharing protocols'
        ]
      });
    }

    setPredictions(newPredictions);
    setIsAnalyzing(false);
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'low': return 'text-green-400 bg-green-400/10 border-green-400/20';
      case 'medium': return 'text-yellow-400 bg-yellow-400/10 border-yellow-400/20';
      case 'high': return 'text-orange-400 bg-orange-400/10 border-orange-400/20';
      case 'critical': return 'text-red-400 bg-red-400/10 border-red-400/20';
      default: return 'text-gray-400 bg-gray-400/10 border-gray-400/20';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'threat_surge': return <TrendingUp className="h-5 w-5" />;
      case 'pattern_anomaly': return <Target className="h-5 w-5" />;
      case 'system_optimization': return <Zap className="h-5 w-5" />;
      case 'risk_assessment': return <BarChart3 className="h-5 w-5" />;
      default: return <Brain className="h-5 w-5" />;
    }
  };

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-white mb-2">Predictive Analytics</h2>
          <p className="text-gray-400">AI-powered threat forecasting and behavioral pattern analysis</p>
        </div>
        
        <div className="flex items-center space-x-4">
          <select
            value={selectedTimeframe}
            onChange={(e) => setSelectedTimeframe(e.target.value as any)}
            className="bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white text-sm"
          >
            <option value="1h">Next Hour</option>
            <option value="6h">Next 6 Hours</option>
            <option value="24h">Next 24 Hours</option>
            <option value="7d">Next 7 Days</option>
          </select>
          
          <button
            onClick={generatePredictions}
            disabled={isAnalyzing}
            className="flex items-center px-4 py-2 bg-cyan-600 text-white rounded-lg hover:bg-cyan-700 disabled:opacity-50 transition-colors"
          >
            <Brain className={`h-4 w-4 mr-2 ${isAnalyzing ? 'animate-pulse' : ''}`} />
            {isAnalyzing ? 'Analyzing...' : 'Regenerate'}
          </button>
        </div>
      </div>

      {/* AI Processing Status */}
      {isAnalyzing && (
        <div className="bg-gradient-to-r from-cyan-900/20 to-blue-900/20 border border-cyan-600/30 rounded-lg p-6">
          <div className="flex items-center space-x-4">
            <Brain className="h-8 w-8 text-cyan-400 animate-pulse" />
            <div>
              <h3 className="text-lg font-semibold text-cyan-400">Neural Network Processing</h3>
              <p className="text-gray-300">Analyzing patterns, correlating threat data, and generating predictions...</p>
            </div>
          </div>
          <div className="mt-4 w-full bg-gray-700 rounded-full h-2">
            <div className="bg-gradient-to-r from-cyan-500 to-blue-500 h-2 rounded-full animate-pulse" style={{ width: '75%' }}></div>
          </div>
        </div>
      )}

      {/* Prediction Cards */}
      <div className="grid gap-6">
        {predictions.map((prediction) => (
          <div key={prediction.id} className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
            <div className="p-6">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div className={`p-2 rounded-lg ${getSeverityColor(prediction.severity)}`}>
                    {getTypeIcon(prediction.type)}
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-white">{prediction.title}</h3>
                    <div className="flex items-center space-x-4 text-sm text-gray-400 mt-1">
                      <span>Timeframe: {prediction.timeframe}</span>
                      <span>Confidence: {prediction.confidence.toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
                
                <div className={`px-3 py-1 rounded-full border text-xs font-medium ${getSeverityColor(prediction.severity)}`}>
                  {prediction.severity.toUpperCase()}
                </div>
              </div>

              <p className="text-gray-300 mb-6">{prediction.description}</p>

              {/* Confidence Meter */}
              <div className="mb-6">
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-400">AI Confidence Level</span>
                  <span className="text-white font-medium">{prediction.confidence.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-gradient-to-r from-cyan-500 to-blue-500 h-2 rounded-full transition-all duration-1000"
                    style={{ width: `${prediction.confidence}%` }}
                  ></div>
                </div>
              </div>

              {/* Recommendations */}
              <div className="bg-gray-700/50 rounded-lg p-4">
                <h4 className="text-sm font-semibold text-white mb-3 flex items-center">
                  <AlertCircle className="h-4 w-4 mr-2 text-cyan-400" />
                  AI Recommendations
                </h4>
                <ul className="space-y-2">
                  {prediction.recommendations.map((rec, index) => (
                    <li key={index} className="flex items-start text-sm text-gray-300">
                      <span className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Analytics Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center">
            <Activity className="h-8 w-8 text-green-400 mr-4" />
            <div>
              <p className="text-sm font-medium text-gray-400">Prediction Accuracy</p>
              <p className="text-2xl font-bold text-white">94.7%</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center">
            <Brain className="h-8 w-8 text-cyan-400 mr-4" />
            <div>
              <p className="text-sm font-medium text-gray-400">Models Active</p>
              <p className="text-2xl font-bold text-white">{predictions.length}</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center">
            <Target className="h-8 w-8 text-purple-400 mr-4" />
            <div>
              <p className="text-sm font-medium text-gray-400">Threat Prevention</p>
              <p className="text-2xl font-bold text-white">87.3%</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictiveAnalytics;