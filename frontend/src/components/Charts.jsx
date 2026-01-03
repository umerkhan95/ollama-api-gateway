import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line, Bar, Doughnut } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

export const LineChart = ({ data, options }) => {
  return <Line data={data} options={options} />;
};

export const BarChart = ({ data, options }) => {
  return <Bar data={data} options={options} />;
};

export const DoughnutChart = ({ data, options }) => {
  return <Doughnut data={data} options={options} />;
};
