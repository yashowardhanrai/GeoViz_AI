import * as THREE from 'three';

/**
 * Creates the custom ShaderMaterial for the Earth surface.
 * Handles textures, colors, and lighting.
 */
export function createEarthMaterial(textures, sunDirection) {
  return new THREE.ShaderMaterial({
    uniforms: {
      uDayTex:      { value: textures.day },
      uNightTex:    { value: textures.night },
      uBumpTex:     { value: textures.bump },
      uSpecularTex: { value: textures.specular },
      uSunDir:      { value: sunDirection }
    },
    vertexShader: EARTH_VERTEX,
    fragmentShader: EARTH_FRAGMENT,
    transparent: false,
    depthWrite: true
  });
}

// ─── Vertex Shader (View-Space Coordinate System) ────────────────────────────

const EARTH_VERTEX = /* glsl */`
  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vViewDir;

  void main() {
    vUv = uv;
    // Normal in view-space: updates dynamically with camera rotation
    vNormal = normalize(normalMatrix * normal);

    // In view space, the vertex position is relative to camera at [0,0,0]
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    vViewDir = normalize(-mvPosition.xyz);

    gl_Position = projectionMatrix * mvPosition;
  }
`;

// ─── Fragment Shader (Vibrant Day, Night City Lights, Cinematic Specular) ─────

const EARTH_FRAGMENT = /* glsl */`
  uniform sampler2D uDayTex;
  uniform sampler2D uNightTex;
  uniform sampler2D uBumpTex;
  uniform sampler2D uSpecularTex;
  uniform vec3 uSunDir; // We'll compute cinematic lighting relative to camera/view space

  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vViewDir;

  void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(vViewDir);

    // Cinematic sun direction offset in view space:
    // Positions the sun slightly to the right, top, and front of the camera.
    vec3 L = normalize(vec3(0.35, 0.25, 0.9));

    // Sample textures
    vec3 dayCol     = texture2D(uDayTex, vUv).rgb;
    vec3 nightCol   = texture2D(uNightTex, vUv).rgb;
    float waterMask = texture2D(uSpecularTex, vUv).r;

    // Boost dayCol for vibrancy
    vec3 dayVibrant = dayCol * 1.55;
    // Blend a premium deep royal blue into the water regions to prevent them from looking like a black void
    dayVibrant = mix(dayVibrant, vec3(0.06, 0.16, 0.38) * 1.6, waterMask * 0.35);

    // ── Day/Night Transition ──
    float cosTheta = dot(N, L);
    // Smooth transition from lit to unlit hemisphere
    float dayMix = smoothstep(-0.25, 0.25, cosTheta);

    // ── Diffuse Shading ──
    // Minimum day brightness is 0.55 so terrain details are always crisp and legible
    float diffuse = max(cosTheta, 0.0) * 0.45 + 0.55;

    // ── Specular Highlights on Oceans ──
    vec3 H = normalize(L + V);
    float specAngle = max(dot(N, H), 0.0);
    float specular = pow(specAngle, 48.0) * waterMask * 0.22;

    // ── Day Color Composition ──
    vec3 dayResult = dayVibrant * diffuse;
    // Add warm tint on sun-facing parts
    dayResult *= vec3(1.02, 1.0, 0.98);
    dayResult += vec3(specular);

    // ── Night Color Composition ──
    // Night lights are bright, oceans have a soft deep blue glow, and continents are clearly visible
    vec3 nightResult = nightCol * 2.5 + dayVibrant * 0.26;

    // ── Composite ──
    vec3 result = mix(nightResult, dayResult, dayMix);

    // Ambient global overlay for clear landmass visibility on the shadow side
    result += dayVibrant * 0.28;

    // Prevent overexposure
    result = min(result, vec3(1.0));

    gl_FragColor = vec4(result, 1.0);
  }
`;
