import { useMemo } from 'react';

/**
 * Standard, clean semi-circular risk gauge.
 * Integrates naturally into the dashboard card layout with solid color segments,
 * high-contrast metallic needle, and clean typography.
 */
export default function RiskGauge({ value = 0, label = 'CURRENT RISK SCORE' }) {
  const clampedValue = Math.max(0, Math.min(100, value));

  // Bloomberg/Palantir style color zones
  const zoneColors = {
    yellow: '#F5C400',
    amber: '#8A6A12',
    red: '#5B1730'
  };

  // Center coordinate mapping
  const cx = 140;
  const cy = 135;
  const radius = 95;
  const strokeWidth = 10;

  // Rotation for the needle (0% -> -90deg, 100% -> +90deg)
  const rotationDeg = ((clampedValue / 100) * 180) - 90;
  // Reduce needle length to 75% of radius (95 * 0.75 ≈ 71)
  const needleLen = 71;

  // Active risk color matching
  const activeColor = useMemo(() => {
    if (clampedValue >= 70) return zoneColors.red;
    if (clampedValue >= 50) return zoneColors.amber;
    return zoneColors.yellow;
  }, [clampedValue]);

  // Math helper to generate arc path coordinates with variable radius
  function getArcPath(startPct, endPct, arcRadius = radius) {
    const a1 = Math.PI - (startPct / 100) * Math.PI;
    const a2 = Math.PI - (endPct / 100) * Math.PI;
    const x1 = cx + arcRadius * Math.cos(a1);
    const y1 = cy - arcRadius * Math.sin(a1);
    const x2 = cx + arcRadius * Math.cos(a2);
    const y2 = cy - arcRadius * Math.sin(a2);
    
    return `M ${x1} ${y1} A ${arcRadius} ${arcRadius} 0 0 1 ${x2} ${y2}`;
  }

  return (
    <div 
      className="gauge-container-flat"
      style={{
        width: '100%',
        maxWidth: '300px',
        margin: '0 auto',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        boxSizing: 'border-box'
      }}
    >
      <svg 
        viewBox="0 0 280 215" 
        width="100%" 
        height="100%"
        style={{ display: 'block', overflow: 'visible' }}
      >
        <defs>
          {/* Active Segment Glow Filter */}
          <filter id="activeGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
          </filter>

          {/* Needle Drop Shadow */}
          <filter id="needleShadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="2" stdDeviation="1.5" floodColor="#000000" floodOpacity="0.5" />
          </filter>

          {/* Needle Metallic Silver Gradient */}
          <linearGradient id="needleGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#F8F9FA" />
            <stop offset="50%" stopColor="#BFC5D2" />
            <stop offset="100%" stopColor="#6B7280" />
          </linearGradient>

          {/* Hub Metallic Radial Gradient */}
          <radialGradient id="hubGradient" cx="50%" cy="50%" r="50%" fx="35%" fy="35%">
            <stop offset="0%" stopColor="#FFFFFF" />
            <stop offset="60%" stopColor="#BFC5D2" />
            <stop offset="100%" stopColor="#4A4F5D" />
          </radialGradient>
        </defs>

        {/* 1. Semicircular Segmented Background Tracks */}
        {/* Yellow segment (0 - 50) */}
        <path
          d={getArcPath(0, 50)}
          fill="none"
          stroke={zoneColors.yellow}
          strokeWidth={strokeWidth}
          opacity="0.2"
        />
        {/* Amber segment (50 - 70) */}
        <path
          d={getArcPath(50, 70)}
          fill="none"
          stroke={zoneColors.amber}
          strokeWidth={strokeWidth}
          opacity="0.2"
        />
        {/* Red segment (70 - 100) */}
        <path
          d={getArcPath(70, 100)}
          fill="none"
          stroke={zoneColors.red}
          strokeWidth={strokeWidth}
          opacity="0.2"
        />

        {/* 2. Active Progress Arc Overlay & Glow */}
        {/* Faint active glow */}
        <path
          d={getArcPath(0, clampedValue)}
          fill="none"
          stroke={activeColor}
          strokeWidth={strokeWidth + 2}
          opacity="0.25"
          filter="url(#activeGlow)"
        />
        {/* Solid active segment */}
        <path
          d={getArcPath(0, clampedValue)}
          fill="none"
          stroke={activeColor}
          strokeWidth={strokeWidth}
          opacity="0.95"
        />

        {/* 3. Subtle Inner & Outer Shadows for Depth */}
        <path
          d={getArcPath(0, 100, 90)}
          fill="none"
          stroke="rgba(0, 0, 0, 0.4)"
          strokeWidth="0.8"
        />
        <path
          d={getArcPath(0, 100, 100)}
          fill="none"
          stroke="rgba(255, 255, 255, 0.06)"
          strokeWidth="0.8"
        />

        {/* 4. Tick Marks & Radial Labels */}
        {[0, 25, 50, 70, 100].map((tick) => {
          const a = Math.PI - (tick / 100) * Math.PI;
          const x1 = cx + (radius + 2) * Math.cos(a);
          const y1 = cy - (radius + 2) * Math.sin(a);
          const x2 = cx + (radius + 7) * Math.cos(a);
          const y2 = cy - (radius + 7) * Math.sin(a);
          const xText = cx + (radius + 17) * Math.cos(a);
          const yText = cy - (radius + 17) * Math.sin(a);

          return (
            <g key={tick}>
              <line
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke="#64748B"
                strokeWidth="1.5"
                opacity="0.4"
              />
              <text
                x={xText}
                y={yText}
                textAnchor="middle"
                dominantBaseline="middle"
                fill="#94A3B8"
                fontSize="9.5"
                fontWeight="700"
                fontFamily="JetBrains Mono, monospace"
                opacity="0.85"
              >
                {tick}
              </text>
            </g>
          );
        })}

        {/* 5. Needle rotated around center-origin with metallic gradient & shadow */}
        <g
          style={{
            transform: `rotate(${rotationDeg}deg)`,
            transformOrigin: '50% 50%',
            transition: 'transform 0.8s cubic-bezier(0.4, 0, 0.2, 1)'
          }}
          filter="url(#needleShadow)"
        >
          {/* Centering bounding box */}
          <rect x="0" y="0" width="280" height="270" fill="none" pointerEvents="none" />

          {/* Symmetrical triangular needle */}
          <polygon
            points={`${cx - 3.2},${cy + 8} ${cx},${cy - needleLen} ${cx + 3.2},${cy + 8}`}
            fill="url(#needleGradient)"
          />

          {/* Specular highlight along left edge */}
          <polygon
            points={`${cx - 3.2},${cy + 8} ${cx},${cy - needleLen} ${cx},${cy + 8}`}
            fill="#FFFFFF"
            opacity="0.22"
          />

          {/* Subtle light glow dot at the very tip */}
          <circle cx={cx} cy={cy - needleLen} r="3" fill="#FFFFFF" opacity="0.3" filter="url(#activeGlow)" />
          <circle cx={cx} cy={cy - needleLen} r="1" fill="#FFFFFF" opacity="0.95" />
        </g>

        {/* 6. Center Hub with metallic radial gradient */}
        <circle cx={cx} cy={cy} r="8.0" fill="url(#hubGradient)" />
        <circle cx={cx} cy={cy} r="3.2" fill="#0A0E1A" />

        {/* 7. Centered Score Readout (moved downward for visual balance) */}
        <text
          x={cx}
          y={cy + 45}
          textAnchor="middle"
          fill="#FFFFFF"
          fontSize="32"
          fontWeight="800"
          fontFamily="JetBrains Mono, monospace"
        >
          {clampedValue.toFixed(1)}
        </text>

        <text
          x={cx}
          y={cy + 60}
          textAnchor="middle"
          fill="#64748B"
          fontSize="8.5"
          fontWeight="700"
          letterSpacing="0.22em"
          fontFamily="Inter, sans-serif"
        >
          {label}
        </text>
      </svg>
    </div>
  );
}
