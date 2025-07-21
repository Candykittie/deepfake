import React, { useState } from 'react';
import { AlertTriangle, Eye, Shield, Clock, Filter, RefreshCw } from 'lucide-react';

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

interface ThreatMonitorProps {
  results: DetectionResult[];
}

const ThreatMonitor: React.FC<ThreatMonitorProps> = ({ results }) => {
  const [filterLevel, setFilterLevel] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'timestamp' | 'confidence' | 'threatLevel'>('timestamp');

  const filteredResults = results
    .filter(result => filterLevel === 'all' || result.threatLevel === filterLevel)
    .filter(result => result.isDeepfake) // Only show threats
    .sort((a, b) => {
      switch (sortBy) {
        case 'timestamp':
          return b.timestamp.getTime() - a.timestamp.getTime();
        case 'confidence':
          return b.confidence - a.confidence;
        case 'threatLevel':
          const levels = { low: 1, medium: 2, high: 3, critical: 4 };
          return levels[b.threatLevel] - levels[a.threatLevel];
        default:
          return 0;
      }
    });

  const getThreatColor = (level: string) => {
    switch (level) {
      case 'low': return 'text-green-400 bg-green-400/10 border-green-400/20';
      case 'medium': return 'text-yellow-400 bg-yellow-400/10 border-yellow-400/20';
      case 'high': return 'text-orange-400 bg-orange-400/10 border-orange-400/20';
      case 'critical': return 'text-red-400 bg-red-400/10 border-red-400/20';
      default: return 'text-gray-400 bg-gray-400/10 border-gray-400/20';
    }
  };

  const criticalThreats = results.filter(r => r.threatLevel === 'critical' && r.isDeepfake).length;
  const highThreats = results.filter(r => r.threatLevel === 'high' && r.isDeepfake).length;
  const totalThreats = results.filter(r => r.isDeepfake).length;

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-white mb-2">Threat Monitor</h2>
        <p className="text-gray-400">Real-time monitoring of detected deepfake threats and security alerts</p>
      </div>

      {/* Threat Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-red-900/20 border border-red-600/30 rounded-lg p-6">
          <div className="flex items-center">
            <AlertTriangle className="h-8 w-8 text-red-400 mr-4" />
            <div>
              <p className="text-sm font-medium text-red-400">Critical Threats</p>
              <p className="text-3xl font-bold text-white">{criticalThreats}</p>
            </div>
          </div>
        </div>

        <div className="bg-orange-900/20 border border-orange-600/30 rounded-lg p-6">
          <div className="flex items-center">
            <Eye className="h-8 w-8 text-orange-400 mr-4" />
            <div>
              <p className="text-sm font-medium text-orange-400">High Risk</p>
              <p className="text-3xl font-bold text-white">{highThreats}</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <div className="flex items-center">
            <Shield className="h-8 w-8 text-cyan-400 mr-4" />
            <div>
              <p className="text-sm font-medium text-gray-400">Total Threats</p>
              <p className="text-3xl font-bold text-white">{totalThreats}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Filter className="h-4 w-4 text-gray-400" />
              <label className="text-sm text-gray-400">Filter by level:</label>
              <select
                value={filterLevel}
                onChange={(e) => setFilterLevel(e.target.value)}
                className="bg-gray-700 border border-gray-600 rounded px-3 py-1 text-white text-sm"
              >
                <option value="all">All Levels</option>
                <option value="critical">Critical</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
              </select>
            </div>

            <div className="flex items-center space-x-2">
              <label className="text-sm text-gray-400">Sort by:</label>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as any)}
                className="bg-gray-700 border border-gray-600 rounded px-3 py-1 text-white text-sm"
              >
                <option value="timestamp">Latest</option>
                <option value="confidence">Confidence</option>
                <option value="threatLevel">Threat Level</option>
              </select>
            </div>
          </div>

          <button className="flex items-center px-4 py-2 bg-cyan-600 text-white rounded-lg hover:bg-cyan-700 transition-colors">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </button>
        </div>
      </div>

      {/* Threat List */}
      <div className="space-y-4">
        {filteredResults.length > 0 ? (
          filteredResults.map((threat) => (
            <div key={threat.id} className="bg-gray-800 rounded-lg border border-gray-700 p-6">
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-4">
                  <div className="flex-shrink-0 mt-1">
                    <AlertTriangle className={`h-6 w-6 ${
                      threat.threatLevel === 'critical' ? 'text-red-400' :
                      threat.threatLevel === 'high' ? 'text-orange-400' :
                      threat.threatLevel === 'medium' ? 'text-yellow-400' : 'text-green-400'
                    }`} />
                  </div>
                  
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <h4 className="text-lg font-semibold text-white">{threat.filename}</h4>
                      <div className={`px-2 py-1 rounded border text-xs font-medium ${getThreatColor(threat.threatLevel)}`}>
                        {threat.threatLevel.toUpperCase()}
                      </div>
                    </div>
                    
                    <div className="grid md:grid-cols-3 gap-4 mb-4">
                      <div>
                        <p className="text-xs text-gray-400">Detection Confidence</p>
                        <p className="text-lg font-bold text-white">{threat.confidence.toFixed(1)}%</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-400">File Type</p>
                        <p className="text-sm text-gray-300 capitalize">{threat.type}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-400">Detected At</p>
                        <p className="text-sm text-gray-300">{threat.timestamp.toLocaleString()}</p>
                      </div>
                    </div>

                    <div className="bg-gray-700/50 rounded p-3">
                      <p className="text-sm text-gray-300 mb-2">
                        <strong className="text-red-400">THREAT IDENTIFIED:</strong> This media contains artificially generated or manipulated content with high confidence.
                      </p>
                      <div className="grid grid-cols-3 gap-2 text-xs">
                        <div>
                          <span className="text-gray-400">Face Detection: </span>
                          <span className="text-white">{threat.analysisDetails.faceDetection.toFixed(1)}%</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Temporal: </span>
                          <span className="text-white">{threat.analysisDetails.temporalConsistency.toFixed(1)}%</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Artifacts: </span>
                          <span className="text-white">{threat.analysisDetails.artifactDetection.toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="flex flex-col space-y-2">
                  <button className="px-4 py-2 bg-red-600 text-white rounded text-sm hover:bg-red-700 transition-colors">
                    Flag Threat
                  </button>
                  <button className="px-4 py-2 bg-gray-600 text-white rounded text-sm hover:bg-gray-700 transition-colors">
                    Details
                  </button>
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-12 text-center">
            <Shield className="h-16 w-16 text-gray-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">No Active Threats</h3>
            <p className="text-gray-400">
              {totalThreats === 0 
                ? "No deepfake threats detected. System is secure." 
                : "All threats have been filtered out based on your current filter settings."
              }
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ThreatMonitor;