import { useMemo } from 'react';
import { createAtmosphereMaterial } from './shaders/atmosphereShader';

/**
 * Atmospheric scattering limb glow layer.
 * Rendered on a slightly larger sphere with BackSide rendering.
 */
export default function Atmosphere() {
  const material = useMemo(() => createAtmosphereMaterial(), []);

  return (
    <mesh>
      <sphereGeometry args={[0.918, 64, 64]} />
      <primitive object={material} attach="material" />
    </mesh>
  );
}
