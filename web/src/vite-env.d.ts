/// <reference types="vite/client" />

declare module "*.onnx?url" {
  const url: string;
  export default url;
}

declare module "*.wasm?url" {
  const url: string;
  export default url;
}
