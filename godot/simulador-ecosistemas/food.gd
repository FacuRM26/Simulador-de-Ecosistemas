extends Sprite2D

var resource_id: String
var animation_time: float = 0.0
var is_water: bool = false

func initialize_vegetation(data: Dictionary):
	resource_id = "veg"
	is_water = false
	z_index = 1
	
	position = Vector2(data.x + data.w / 2.0, data.y + data.h / 2.0)
	
	var veg_texture = load("res://sprites/vegetacion/prueba_veg.png")
	if veg_texture:
		texture = veg_texture
		modulate = Color.GREEN
		# Escalar para que coincida con el tamaño en píxeles de Python
		var tex_width = texture.get_width()
		var tex_height = texture.get_height()
		scale = Vector2(data.w / float(tex_width), data.h / float(tex_height))
	else:
		# Fallback: crear textura simple
		_create_rect_sprite(int(data.w), int(data.h))
		modulate = Color.GREEN

func initialize_water(data: Dictionary):
	resource_id = "water"
	is_water = true
	z_index = 1
	
	position = Vector2(data.x + data.w / 2.0, data.y + data.h / 2.0)
	
	# Cargar textura PNG
	var water_texture = load("res://sprites/agua/agua_base.png")
	if water_texture:
		texture = water_texture
		modulate = Color(1, 1, 1, 0.8)
		var tex_width = texture.get_width()
		var tex_height = texture.get_height()
		scale = Vector2(data.w / float(tex_width), data.h / float(tex_height))
	else:
		# Fallback
		_create_rect_sprite(int(data.w), int(data.h))
		modulate = Color(0.3, 0.5, 0.8, 0.7)
	
	# Iniciar animación
	set_process(true)

func _create_rect_sprite(width: int, height: int):
	var img = Image.create(width, height, false, Image.FORMAT_RGBA8)
	img.fill(Color.WHITE)
	texture = ImageTexture.create_from_image(img)
	scale = Vector2(1.0, 1.0)  # Sin escala adicional

func _process(delta):
	if not is_water:
		set_process(false)
		return
	
	animation_time += delta
	
	# Solo pulsación de opacidad (muy simple)
	var pulse = sin(animation_time * 2.0) * 0.1 + 0.9
	modulate.a = 0.8 * pulse
