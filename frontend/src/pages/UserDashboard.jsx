import React, { useState, useEffect } from 'react';
import { apiService } from '../services/api';
import StatsCard from '../components/StatsCard';
import { LineChart, BarChart } from '../components/Charts';
import { Activity, Clock, TrendingUp, Zap } from 'lucide-react';
import { format } from 'date-fns';

const UserDashboard = () => {
  const [stats, setStats] = useState(null);
  const [detailedStats, setDetailedStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    setLoading(true);
    const [statsResult, detailedResult] = await Promise.all([
      apiService.getStats(),
      apiService.getDetailedStats()
    ]);

    if (statsResult.success) {
      setStats(statsResult.data);
    }
    if (detailedResult.success) {
      setDetailedStats(detailedResult.data);
    }
    setLoading(false);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-gray-600 dark:text-gray-400">Loading...</div>
      </div>
    );
  }

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        labels: {
          color: document.documentElement.classList.contains('dark') ? '#e5e7eb' : '#374151'
        }
      }
    },
    scales: {
      y: {
        ticks: {
          color: document.documentElement.classList.contains('dark') ? '#9ca3af' : '#6b7280'
        },
        grid: {
          color: document.documentElement.classList.contains('dark') ? '#374151' : '#e5e7eb'
        }
      },
      x: {
        ticks: {
          color: document.documentElement.classList.contains('dark') ? '#9ca3af' : '#6b7280'
        },
        grid: {
          color: document.documentElement.classList.contains('dark') ? '#374151' : '#e5e7eb'
        }
      }
    }
  };

  const requestsOverTimeData = detailedStats?.requests_by_hour ? {
    labels: Object.keys(detailedStats.requests_by_hour).map(hour => `${hour}:00`),
    datasets: [
      {
        label: 'Requests',
        data: Object.values(detailedStats.requests_by_hour),
        borderColor: 'rgb(14, 165, 233)',
        backgroundColor: 'rgba(14, 165, 233, 0.1)',
        tension: 0.4
      }
    ]
  } : null;

  const endpointData = detailedStats?.requests_by_endpoint ? {
    labels: Object.keys(detailedStats.requests_by_endpoint),
    datasets: [
      {
        label: 'Requests by Endpoint',
        data: Object.values(detailedStats.requests_by_endpoint),
        backgroundColor: [
          'rgba(14, 165, 233, 0.8)',
          'rgba(34, 197, 94, 0.8)',
          'rgba(251, 191, 36, 0.8)',
          'rgba(239, 68, 68, 0.8)',
          'rgba(168, 85, 247, 0.8)',
        ],
      }
    ]
  } : null;

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">My Dashboard</h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">View your API usage statistics</p>
        </div>

        {detailedStats && (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              <StatsCard
                title="Total Requests (24h)"
                value={detailedStats.total_requests_24h}
                icon={Activity}
                description="Last 24 hours"
              />
              <StatsCard
                title="Total Requests (7d)"
                value={detailedStats.total_requests_7d}
                icon={TrendingUp}
                description="Last 7 days"
              />
              <StatsCard
                title="Avg Response Time"
                value={`${(detailedStats?.avg_response_time || 0).toFixed(2)}s`}
                icon={Clock}
                description="Average across all requests"
              />
              <StatsCard
                title="Rate Limit Usage"
                value={`${(detailedStats?.rate_limit_usage_percent || 0).toFixed(1)}%`}
                icon={Zap}
                description={`${detailedStats?.requests_in_current_hour || 0}/${detailedStats?.rate_limit || 100} this hour`}
              />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Response Time Metrics
                </h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Minimum:</span>
                    <span className="font-medium text-gray-900 dark:text-white">
                      {(detailedStats?.min_response_time || 0).toFixed(2)}s
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Maximum:</span>
                    <span className="font-medium text-gray-900 dark:text-white">
                      {(detailedStats?.max_response_time || 0).toFixed(2)}s
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Average:</span>
                    <span className="font-medium text-gray-900 dark:text-white">
                      {(detailedStats?.avg_response_time || 0).toFixed(2)}s
                    </span>
                  </div>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Usage Summary
                </h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Successful Requests:</span>
                    <span className="font-medium text-green-600 dark:text-green-400">
                      {detailedStats.successful_requests_24h}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Failed Requests:</span>
                    <span className="font-medium text-red-600 dark:text-red-400">
                      {detailedStats.failed_requests_24h}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Success Rate:</span>
                    <span className="font-medium text-gray-900 dark:text-white">
                      {((detailedStats.successful_requests_24h / detailedStats.total_requests_24h) * 100 || 0).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {requestsOverTimeData && (
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Requests Over Time (Last 24 Hours)
                </h3>
                <LineChart data={requestsOverTimeData} options={chartOptions} />
              </div>
            )}

            {endpointData && (
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Requests by Endpoint
                </h3>
                <BarChart data={endpointData} options={chartOptions} />
              </div>
            )}
          </>
        )}

        {!detailedStats && (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-8 text-center">
            <p className="text-gray-600 dark:text-gray-400">No usage data available yet</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default UserDashboard;
