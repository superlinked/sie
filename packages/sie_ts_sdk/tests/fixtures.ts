export const MINIMAL_JPEG_BASE64 =
  "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAVEAEBAAAAAAAAAAAAAAAAAAAAAf/aAAwDAQACEAMQAAAB6A//xAAVEAEBAAAAAAAAAAAAAAAAAAABAP/aAAgBAQABPwCf/8QAFBEBAAAAAAAAAAAAAAAAAAAAEP/aAAgBAgEBPwCf/8QAFBEBAAAAAAAAAAAAAAAAAAAAEP/aAAgBAwEBPwCf/9k=";

export const MINIMAL_JPEG_BYTES = Uint8Array.from(atob(MINIMAL_JPEG_BASE64), (character) =>
  character.charCodeAt(0),
);
