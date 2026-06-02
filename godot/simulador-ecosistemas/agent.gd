extends Sprite2D

const AGENT_SIZE_PX = 50

var agent_id: String
var energy_food: float
var energy_water: float
var status_circle: Sprite2D
var last_position: Vector2 # Para guardar posición anterior
var is_predator: bool = false
#var base_scale: Vector2 = Vector2(0.25, 0.25)  # Escala base para los sprites

func initialize(data: Dictionary):
	agent_id = str(data.id)
	is_predator = (data.role == 1)
	
	# Cargar sprite de pixel art
	if is_predator:
		texture = load("res://sprites/depredador/depredador1.png")
	else:
		texture = load("res://sprites/presa/presa1.png")
	
	# Escala fija para el sprite principal
	#scale = base_scale
	
	if texture:
		var sprite_width = texture.get_width()
		var sprite_height = texture.get_height()
		# Escalar para que el sprite tenga el tamaño correcto
		scale = Vector2(
			AGENT_SIZE_PX / float(sprite_width),
			AGENT_SIZE_PX / float(sprite_height)
		)
	
	# Guardar posición inicial
	last_position = Vector2(data.x, data.y)
	
	# Crear el círculo PRIMERO antes de actualizar datos
	_create_status_circle()
	update_data(data)

func update_data(data: Dictionary):
	energy_food = data.food
	energy_water = data.water
	
	var new_position = Vector2(data.x, data.y)
	
	# Determinar dirección del movimiento y ajustar orientación
	if new_position != last_position:
		_update_direction(new_position)
	
	position = new_position
	#position = Vector2(data.x, data.y)
	
	# Actualizar color del círculo de estado
	var energy_avg = (energy_food + energy_water) / 2.0
	_update_status_color(energy_avg)
	
	# Guardar posición actual para la próxima comparación
	last_position = new_position

func _update_direction(new_position: Vector2):
	# Calcular dirección del movimiento
	var direction = new_position.x - last_position.x
	
	# Si se mueve hacia la izquierda, voltear horizontalmente
	if direction < 0:
		scale.x = -abs(scale.x)  # Escala negativa para mirar izquierda
	elif direction > 0:
		scale.x = abs(scale.x)   # Escala positiva para mirar derecha
	# Si direction == 0, mantener la dirección actual

func _create_status_circle():
	status_circle = Sprite2D.new()
	
	# Crear textura circular
	var img = Image.create(16, 16, false, Image.FORMAT_RGBA8)
	img.fill(Color.TRANSPARENT)
	
	for x in range(16):
		for y in range(16):
			var dx = x - 8
			var dy = y - 8
			if dx*dx + dy*dy <= 64:  # Radio 8
				img.set_pixel(x, y, Color.WHITE)
	
	var circle_texture = ImageTexture.create_from_image(img)
	status_circle.texture = circle_texture
	
	#status_circle.position = Vector2(0, -130)  # Posición encima del agente
	#status_circle.scale = Vector2(0.8, 0.8)   # Escala fija para el círculo
	status_circle.position = Vector2(0, -AGENT_SIZE_PX - 5)
	status_circle.scale = Vector2(0.6, 0.6)
	status_circle.top_level = true # Asegurar que el círculo siempre mire hacia la derecha
	
	add_child(status_circle)

func _update_status_color(energy: float):
	# Verificar que status_circle existe antes de modificar
	if status_circle == null:
		return
	
	var status_color: Color
	
	if energy > 75:
		status_color = Color.GREEN
	elif energy > 50:
		status_color = Color.YELLOW
	elif energy > 25:
		status_color = Color.ORANGE
	else:
		status_color = Color.RED
	
	status_circle.modulate = status_color
	
	# Actualizar posición del círculo para que siga al agente
	#status_circle.global_position = global_position + Vector2(0, -40)
	status_circle.global_position = global_position + Vector2(0, -AGENT_SIZE_PX - 5)
