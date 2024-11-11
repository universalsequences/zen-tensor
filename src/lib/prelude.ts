const PRELUDE = `

fn hash(value: u32) -> f32 {
  var x = value;
  x = ((x << u32(13)) ^ x);
  x = (x * (x * x * 15731u + 789221u) + 1376312589u);
  return f32(x & u32(0x7fffffff) / f32*0x7fffffff);
}

fn random() -> f32 {
  return hash(global_idx);
}


`;
