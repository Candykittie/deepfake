import React, { useState } from 'react';
import { Database, Download, Filter, Search, Clock, User, FileText, Shield } from 'lucide-react';

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

interface AuditLogProps {
  results: DetectionResult[];
}

const AuditLog: React.FC<AuditLogProps> = ({ results }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState<'all' | 'image' | 'video'>('all');
  const [filterResult, setFilterResult] = useState<'all' | 'deepfake' | 'authentic'>('all');

  const filteredLogs = results
    .filter(result => 
      result.filename.toLowerCase().includes(searchTerm.toLowerCase())
    )
    .filter(result => 
      filterType === 'all' || result.type === filterType
    )
    .filter(result => {
      if (filterResult === 'all') return true;
      if (filterResult === 'deepfake') return result.isDeepfake;
      return !result.isDeepfake;
    })
    .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());

  const exportLog = () => {
    const logData = filteredLogs.map(result => ({
      timestamp: result.timestamp.toISOString(),
      filename: result.filename,
      type: result.type,
      isDeepfake: result.isDeepfake,
      confidence: result.confidence,
      threatLevel: result.threatLevel,
      faceDetection: result.analysisDetails.faceDetection,
      temporalConsistency: result.analysisDetails.temporalConsistency,
      artifactDetection: result.analysisDetails.artifactDetection,
    }));

    const blob = new Blob([JSON.stringify(logData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `deepfake-audit-log-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-white mb-2">Audit Log</h2>
        <p className="text-gray-400">Comprehensive logging of all detection activities and security events</p>
      </div>

      {/* Controls */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-4">
            {/* Search */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search files..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-cyan-400"
              />
            </div>

            {/* Filters */}
            <div className="flex items-center space-x-2">
              <Filter className="h-4 w-4 text-gray-400" />
              <select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value as any)}
                className="bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white text-sm"
              >
                <option value="all">All Types</option>
                <option value="image">Images</option>
                <option value="video">Videos</option>
              </select>
            </div>

            <select
              value={filterResult}
              onChange={(e) => setFilterResult(e.target.value as any)}
              className="bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white text-sm"
            >
              <option value="all">All Results</option>
              <option value="deepfake">Deepfakes</option>
              <option value="authentic">Authentic</option>
            </select>
          </div>

          <button
            onClick={exportLog}
            className="flex items-center px-4 py-2 bg-cyan-600 text-white rounded-lg hover:bg-cyan-700 transition-colors"
          >
            <Download className="h-4 w-4 mr-2" />
            Export Log
          </button>
        </div>

        <div className="mt-4 flex items-center justify-between text-sm text-gray-400">
          <span>Showing {filteredLogs.length} of {results.length} entries</span>
          <span>Last updated: {new Date().toLocaleString()}</span>
        </div>
      </div>

      {/* Log Table */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
        {filteredLogs.length > 0 ? (
          <>
            {/* Table Header */}
            <div className="bg-gray-700 px-6 py-4 border-b border-gray-600">
              <div className="grid grid-cols-12 gap-4 text-sm font-medium text-gray-300">
                <div className="col-span-3">File Name</div>
                <div className="col-span-2">Timestamp</div>
                <div className="col-span-1">Type</div>
                <div className="col-span-2">Result</div>
                <div className="col-span-2">Confidence</div>
                <div className="col-span-2">Threat Level</div>
              </div>
            </div>

            {/* Table Body */}
            <div className="divide-y divide-gray-700">
              {filteredLogs.map((entry) => (
                <div key={entry.id} className="px-6 py-4 hover:bg-gray-700/50 transition-colors">
                  <div className="grid grid-cols-12 gap-4 items-center text-sm">
                    <div className="col-span-3">
                      <div className="flex items-center space-x-2">
                        <FileText className="h-4 w-4 text-gray-400" />
                        <span className="text-white font-medium truncate">{entry.filename}</span>
                      </div>
                    </div>
                    
                    <div className="col-span-2">
                      <div className="flex items-center space-x-1 text-gray-300">
                        <Clock className="h-3 w-3" />
                        <span>{entry.timestamp.toLocaleString()}</span>
                      </div>
                    </div>
                    
                    <div className="col-span-1">
                      <span className="px-2 py-1 bg-gray-600 text-gray-300 rounded text-xs uppercase">
                        {entry.type}
                      </span>
                    </div>
                    
                    <div className="col-span-2">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        entry.isDeepfake
                          ? 'bg-red-900/50 text-red-400'
                          : 'bg-green-900/50 text-green-400'
                      }`}>
                        {entry.isDeepfake ? 'DEEPFAKE' : 'AUTHENTIC'}
                      </span>
                    </div>
                    
                    <div className="col-span-2">
                      <div className="flex items-center space-x-2">
                        <div className="flex-1 bg-gray-600 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${
                              entry.isDeepfake ? 'bg-red-500' : 'bg-green-500'
                            }`}
                            style={{ width: `${entry.confidence}%` }}
                          ></div>
                        </div>
                        <span className="text-white text-xs w-10 text-right">
                          {entry.confidence.toFixed(0)}%
                        </span>
                      </div>
                    </div>
                    
                    <div className="col-span-2">
                      <span className={`px-2 py-1 rounded text-xs font-medium border ${
                        entry.threatLevel === 'critical' ? 'bg-red-400/10 text-red-400 border-red-400/20' :
                        entry.threatLevel === 'high' ? 'bg-orange-400/10 text-orange-400 border-orange-400/20' :
                        entry.threatLevel === 'medium' ? 'bg-yellow-400/10 text-yellow-400 border-yellow-400/20' :
                        'bg-green-400/10 text-green-400 border-green-400/20'
                      }`}>
                        {entry.threatLevel.toUpperCase()}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </>
        ) : (
          <div className="p-12 text-center">
            <Database className="h-16 w-16 text-gray-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">No Log Entries</h3>
            <p className="text-gray-400">
              {results.length === 0 
                ? "No detection activities to log yet." 
                : "No entries match your current filter criteria."
              }
            </p>
          </div>
        )}
      </div>

      {/* Log Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center">
            <FileText className="h-8 w-8 text-cyan-400 mr-3" />
            <div>
              <p className="text-sm font-medium text-gray-400">Total Entries</p>
              <p className="text-xl font-bold text-white">{results.length}</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center">
            <Shield className="h-8 w-8 text-red-400 mr-3" />
            <div>
              <p className="text-sm font-medium text-gray-400">Threats Logged</p>
              <p className="text-xl font-bold text-white">
                {results.filter(r => r.isDeepfake).length}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center">
            <User className="h-8 w-8 text-green-400 mr-3" />
            <div>
              <p className="text-sm font-medium text-gray-400">Authentic Files</p>
              <p className="text-xl font-bold text-white">
                {results.filter(r => !r.isDeepfake).length}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center">
            <Clock className="h-8 w-8 text-yellow-400 mr-3" />
            <div>
              <p className="text-sm font-medium text-gray-400">Last Activity</p>
              <p className="text-sm font-bold text-white">
                {results.length > 0 
                  ? results[0].timestamp.toLocaleString()
                  : 'No activity'
                }
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AuditLog;