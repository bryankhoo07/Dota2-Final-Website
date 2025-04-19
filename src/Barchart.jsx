import React from "react";
import {
  BarChart as RechartsBarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from "recharts";

const WinProbabilityChart = ({ winProbability }) => {
  if (winProbability === null) return null;

  const data = [
    { 
      name: "Teams", 
      radiant: winProbability * 100, 
      dire: (1 - winProbability) * 100 
    }
  ];



  return (
    <ResponsiveContainer width={500} height={250}>
      <RechartsBarChart data={data}>

        <CartesianGrid strokeDasharray="4 4" />
        <XAxis dataKey="name"/>
        <YAxis domain={[0, 100]} />
        <Tooltip />
        <Bar dataKey="radiant"  fill="#22c55e" barSize={70} name="Radiant"/>
        <Bar dataKey="dire" fill="#ef4444" barSize={70} name="Dire"/>

        
        
      </RechartsBarChart>
    </ResponsiveContainer>
  );
};

export default WinProbabilityChart;
