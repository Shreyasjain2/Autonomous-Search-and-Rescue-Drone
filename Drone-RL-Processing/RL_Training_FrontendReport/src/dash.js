import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer
} from 'recharts';
import { Map, Maximize2, Users, Target, AlertTriangle, Award, Activity } from 'lucide-react';
import { 
  Box, 
  Card, 
  CardContent, 
  Typography, 
  Grid, 
  Alert, 
  AlertTitle,
  Paper,
  ThemeProvider,
  createTheme,
  useTheme,
  LinearProgress,
  IconButton,
  Chip
} from '@mui/material';
import { blue, green, orange, red, grey } from '@mui/material/colors';

// Create a dark theme
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: blue,
    secondary: green,
    background: {
      default: '#1a1a1a',
      paper: '#242424'
    }
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif'
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          borderRadius: 12,
          border: '1px solid rgba(255, 255, 255, 0.12)'
        }
      }
    }
  }
});

const generateTrainingData = (timestep) => {
  // ... (keep the same data generation logic from previous version)
  const episodeBase = Math.min(1 - Math.exp(-timestep / 1000), 0.85);
  const learningNoise = () => (Math.random() - 0.5) * 0.1;

  const rewards = Array(20).fill(0).map((_, i) => {
    const step = timestep - 19 + i;
    const progress = Math.min(1 - Math.exp(-step / 1000), 0.85);
    
    return {
      step,
      total: 30 * progress + learningNoise() * 20,
      coverage: 15 * progress + learningNoise() * 10,
      detection: 8 * progress + learningNoise() * 5,
      cohesion: 5 * progress + learningNoise() * 3
    };
  });

  const metrics = {
    episodeLength: 3000 + Math.floor(2000 * (1 - episodeBase)),
    victimsFound: Math.floor(5 + 10 * episodeBase),
    coverage: 40 + 50 * episodeBase,
    totalReward: 2000 + 3000 * episodeBase,
    policyLoss: 0.5 * Math.exp(-timestep / 500) + 0.05,
    valueLoss: 0.8 * Math.exp(-timestep / 400) + 0.1
  };

  const drones = Array(4).fill(0).map((_, i) => {
    const angle = (i * Math.PI / 2) + (timestep * 0.01);
    const radius = 10 * (1 - Math.exp(-timestep / 500));
    
    return {
      id: i + 1,
      x: Math.sin(angle) * radius,
      y: Math.cos(angle) * radius,
      z: 2.0 + Math.sin(timestep * 0.01) * 0.2,
      battery: 95 - (timestep % 100) * 0.15,
      status: Math.random() > 0.98 ? 'Warning' : 'Nominal'
    };
  });

  return { rewards, metrics, drones };
};

const DroneStatusCard = ({ drone }) => {
  const theme = useTheme();
  const batteryColor = drone.battery > 90 
    ? green[500] 
    : drone.battery > 70 
    ? orange[500] 
    : red[500];

  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6" component="div">
            Drone {drone.id}
          </Typography>
          <Chip 
            label={drone.status}
            color={drone.status === 'Nominal' ? 'success' : 'error'}
            size="small"
          />
        </Box>
        <Box mb={2}>
          <Typography variant="body2" color="textSecondary">Position</Typography>
          <Typography variant="body1">
            ({drone.x.toFixed(1)}, {drone.y.toFixed(1)}, {drone.z.toFixed(1)})
          </Typography>
        </Box>
        <Box>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
            <Typography variant="body2" color="textSecondary">Battery</Typography>
            <Typography variant="body2" style={{ color: batteryColor }}>
              {drone.battery.toFixed(1)}%
            </Typography>
          </Box>
          <LinearProgress 
            variant="determinate" 
            value={drone.battery} 
            sx={{ 
              height: 8, 
              borderRadius: 4,
              backgroundColor: grey[800],
              '& .MuiLinearProgress-bar': {
                backgroundColor: batteryColor
              }
            }}
          />
        </Box>
      </CardContent>
    </Card>
  );
};

const MetricCard = ({ icon: Icon, title, value, trend }) => (
  <Card>
    <CardContent>
      <Box display="flex" alignItems="center" mb={2}>
        <Icon size={20} style={{ marginRight: 8 }} />
        <Typography variant="h6" component="div">
          {title}
        </Typography>
      </Box>
      <Typography variant="h4" component="div" gutterBottom>
        {value}
      </Typography>
      {trend && (
        <Typography 
          variant="body2" 
          color={trend > 0 ? green[500] : red[500]}
        >
          {trend > 0 ? '+' : ''}{trend}% from last episode
        </Typography>
      )}
    </CardContent>
  </Card>
);

const RewardChart = ({ data }) => (
  <Card>
    <CardContent>
      <Typography variant="h6" gutterBottom>
        Training Rewards
      </Typography>
      <Box height={350}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke={grey[800]} />
            <XAxis 
              dataKey="step" 
              stroke={grey[400]}
            />
            <YAxis stroke={grey[400]} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#242424',
                border: '1px solid rgba(255, 255, 255, 0.12)'
              }}
            />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="total" 
              stroke={blue[400]} 
              name="Total Reward"
              strokeWidth={2}
            />
            <Line 
              type="monotone" 
              dataKey="coverage" 
              stroke={green[400]} 
              name="Coverage Reward"
              strokeWidth={2}
            />
            <Line 
              type="monotone" 
              dataKey="detection" 
              stroke={orange[400]} 
              name="Detection Reward"
              strokeWidth={2}
            />
            <Line 
              type="monotone" 
              dataKey="cohesion" 
              stroke={red[400]} 
              name="Cohesion Reward"
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </Box>
    </CardContent>
  </Card>
);

const Dashboard = () => {
  const [timestep, setTimestep] = useState(0);
  const [data, setData] = useState(generateTrainingData(0));

  useEffect(() => {
    const interval = setInterval(() => {
      setTimestep(t => t + 1);
      setData(generateTrainingData(timestep));
    }, 1000);
    return () => clearInterval(interval);
  }, [timestep]);

  return (
    <ThemeProvider theme={darkTheme}>
      <Box 
        sx={{ 
          minHeight: '100vh',
          bgcolor: 'background.default',
          color: 'text.primary',
          p: 3
        }}
      >
        <Box mb={4}>
          <Typography variant="h3" gutterBottom>
            Multi-Drone Search and Rescue Training
          </Typography>
          <Typography variant="h6" color="textSecondary">
            Training Step: {timestep}
          </Typography>
        </Box>

        {data.drones.some(d => d.status === 'Warning') && (
          <Alert 
            severity="error" 
            sx={{ mb: 3 }}
          >
            <AlertTitle>Warning</AlertTitle>
            One or more drones reporting warning status. Check individual drone statuses below.
          </Alert>
        )}

        <Grid container spacing={3}>
          <Grid item xs={3}>
            <MetricCard 
              icon={Activity} 
              title="Episode Length" 
              value={data.metrics.episodeLength} 
              trend={-2.5} 
            />
          </Grid>
          <Grid item xs={3}>
            <MetricCard 
              icon={Target} 
              title="Victims Found" 
              value={data.metrics.victimsFound} 
              trend={5.3} 
            />
          </Grid>
          <Grid item xs={3}>
            <MetricCard 
              icon={Map} 
              title="Area Coverage" 
              value={`${data.metrics.coverage.toFixed(1)}%`} 
              trend={3.2} 
            />
          </Grid>
          <Grid item xs={3}>
            <MetricCard 
              icon={Award} 
              title="Total Reward" 
              value={data.metrics.totalReward.toFixed(0)} 
              trend={3.8} 
            />
          </Grid>

          <Grid item xs={8}>
            <RewardChart data={data.rewards} />
          </Grid>
          <Grid item xs={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Training Metrics
                </Typography>
                <Box mt={2}>
                  <Typography variant="body2" color="textSecondary" gutterBottom>
                    Policy Loss
                  </Typography>
                  <Typography variant="h5" gutterBottom>
                    {data.metrics.policyLoss.toFixed(3)}
                  </Typography>
                  <Typography variant="body2" color="textSecondary" gutterBottom>
                    Value Loss
                  </Typography>
                  <Typography variant="h5">
                    {data.metrics.valueLoss.toFixed(3)}
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {data.drones.map(drone => (
            <Grid item xs={3} key={drone.id}>
              <DroneStatusCard drone={drone} />
            </Grid>
          ))}
        </Grid>
      </Box>
    </ThemeProvider>
  );
};

export default Dashboard;