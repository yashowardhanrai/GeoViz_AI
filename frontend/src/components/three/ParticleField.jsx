import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';

function Particles({ count = 400 }) {
  const mesh = useRef();
  const linesMesh = useRef();

  const [positions, velocities] = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const vel = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      pos[i * 3]     = (Math.random() - 0.5) * 50;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 50;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 20;
      vel[i * 3]     = (Math.random() - 0.5) * 0.003;
      vel[i * 3 + 1] = (Math.random() - 0.5) * 0.003;
      vel[i * 3 + 2] = (Math.random() - 0.5) * 0.001;
    }
    return [pos, vel];
  }, [count]);

  const linePositions = useMemo(() => new Float32Array(count * count * 6), [count]);

  useFrame(() => {
    if (!mesh.current) return;
    const posArray = mesh.current.geometry.attributes.position.array;

    // Move particles
    for (let i = 0; i < count * 3; i++) {
      posArray[i] += velocities[i];
      // Wrap around
      if (posArray[i] > 25) posArray[i] = -25;
      if (posArray[i] < -25) posArray[i] = 25;
    }
    mesh.current.geometry.attributes.position.needsUpdate = true;

    // Build connection lines
    if (!linesMesh.current) return;
    let lineIdx = 0;
    const maxDist = 3.5;
    for (let i = 0; i < count && lineIdx < linePositions.length - 6; i++) {
      for (let j = i + 1; j < count && lineIdx < linePositions.length - 6; j++) {
        const dx = posArray[i * 3] - posArray[j * 3];
        const dy = posArray[i * 3 + 1] - posArray[j * 3 + 1];
        const dz = posArray[i * 3 + 2] - posArray[j * 3 + 2];
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < maxDist) {
          linePositions[lineIdx++] = posArray[i * 3];
          linePositions[lineIdx++] = posArray[i * 3 + 1];
          linePositions[lineIdx++] = posArray[i * 3 + 2];
          linePositions[lineIdx++] = posArray[j * 3];
          linePositions[lineIdx++] = posArray[j * 3 + 1];
          linePositions[lineIdx++] = posArray[j * 3 + 2];
        }
      }
    }
    // Fill remaining with zeros
    for (let i = lineIdx; i < linePositions.length; i++) {
      linePositions[i] = 0;
    }
    linesMesh.current.geometry.attributes.position.needsUpdate = true;
    linesMesh.current.geometry.setDrawRange(0, lineIdx / 3);
  });

  return (
    <>
      <points ref={mesh}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={count}
            array={positions}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.06}
          color="#00f0ff"
          transparent
          opacity={0.5}
          sizeAttenuation
          depthWrite={false}
        />
      </points>

      <lineSegments ref={linesMesh}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={linePositions.length / 3}
            array={linePositions}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial
          color="#00f0ff"
          transparent
          opacity={0.06}
          depthWrite={false}
        />
      </lineSegments>
    </>
  );
}

export default function ParticleField() {
  return (
    <div className="three-bg-container">
      <Canvas
        camera={{ position: [0, 0, 15], fov: 60 }}
        style={{ background: 'transparent' }}
        gl={{ alpha: true, antialias: false }}
        dpr={[1, 1.5]}
      >
        <Particles />
      </Canvas>
    </div>
  );
}
