extends Node2D

const API_URL = "http://127.0.0.1:5000/state"
const UPDATE_INTERVAL = 0.07

const MAP_WIDTH = 800
const MAP_HEIGHT = 600
const AGENT_SIZE_PX = 30  # Debe coincidir con Specie.AGENT_SIZE en Python

var agent_scene = preload("res://Agent.tscn")
var food_scene = preload("res://Food.tscn")

var http_request: HTTPRequest
var update_timer: Timer
var agents_container: Node2D
var food_container: Node2D

var active_agents = {}
var active_vegetation = {}
var active_water = {}

@onready var stats_label = $UI/StatsLabel

func _ready():
	# Ajustar tamaño de ventana para coincidir con el mapa Python
	get_window().size = Vector2i(MAP_WIDTH, MAP_HEIGHT + 200)  # +200 para stats
	
	# Configurar color de fondo cafecito tierra
	RenderingServer.set_default_clear_color(Color(0.627, 0.515, 0.391, 1.0))
	
	agents_container = Node2D.new()
	agents_container.name = "Agents"
	agents_container.z_index = 3 # Para que se muestren encima
	add_child(agents_container)
	
	food_container = Node2D.new()
	food_container.name = "Resources"
	food_container.z_index = 1 # Para que se muestren detrás de los agentes
	add_child(food_container)
	
	http_request = HTTPRequest.new()
	add_child(http_request)
	http_request.timeout = 5.0
	http_request.request_completed.connect(_on_request_completed)
	
	update_timer = Timer.new()
	update_timer.wait_time = UPDATE_INTERVAL
	update_timer.autostart = true
	update_timer.timeout.connect(_fetch_state)
	add_child(update_timer)
	
	await get_tree().create_timer(1.0).timeout
	_fetch_state()

func _fetch_state():
	if http_request.get_http_client_status() == HTTPClient.STATUS_REQUESTING:
		return
	
	var error = http_request.request(API_URL)
	if error != OK:
		print("Error al hacer request: ", error)

func _on_request_completed(result, response_code, headers, body):
	if result != HTTPRequest.RESULT_SUCCESS:
		print("Error HTTP: ", result)
		return
	
	if response_code != 200:
		print("Código: ", response_code)
		return
	
	var body_string = body.get_string_from_utf8()
	var json = JSON.new()
	var parse_result = json.parse(body_string)
	
	if parse_result != OK:
		print("Error parseando JSON")
		return
	
	var data = json.data
	
	if not data.has("agents"):
		print("Sin agentes en datos")
		return
	
	# print("Actualizando ecosistema - Agentes: ", data.agents.size())
	_update_ecosystem(data)

func _update_ecosystem(data):
	# Actualizar vegetación si existe
	if data.has("vegetation"):
		_update_vegetation(data.vegetation)
	
	# Actualizar fuentes de agua si existe
	if data.has("water_sources"):
		_update_water(data.water_sources)
	
	_update_stats(data)
	_update_agents(data.agents)

func _update_agents(agents_data):
	var current_agents = {}
	
	for agent_data in agents_data:
		if not agent_data.alive:
			continue
		
		var agent_id = str(agent_data.id)
		current_agents[agent_id] = true
		
		if active_agents.has(agent_id):
			active_agents[agent_id].update_data(agent_data)
		else:
			var agent = agent_scene.instantiate()
			agent.initialize(agent_data)
			agents_container.add_child(agent)
			active_agents[agent_id] = agent
			print("Agente creado: ", agent_id, " en pos: ", agent.position)
	
	# Eliminar agentes muertos
	for agent_id in active_agents.keys():
		if not current_agents.has(agent_id):
			active_agents[agent_id].queue_free()
			active_agents.erase(agent_id)

func _update_vegetation(veg_data):
	var current_veg = {}
	
	for veg in veg_data:
		# ← Usar el ID estable que viene de Python, no el índice del array
		var veg_id = "veg_" + str(veg.id)
		current_veg[veg_id] = true
		
		if not active_vegetation.has(veg_id):
			var food = food_scene.instantiate()
			food.initialize_vegetation(veg)
			food_container.add_child(food)
			active_vegetation[veg_id] = food
		
		# Si ya existe, podrías actualizar charges aquí (opcionalmente)
	
	# Limpiar vegetación que ya no existe (fue consumida)
	for veg_id in active_vegetation.keys():
		if not current_veg.has(veg_id):
			active_vegetation[veg_id].queue_free()
			active_vegetation.erase(veg_id)

func _update_water(water_data):
	var current_water = {}
	
	for water in water_data:
		# ← Usar el ID estable que viene de Python
		var water_id = "water_" + str(water.id)
		current_water[water_id] = true
		
		if not active_water.has(water_id):
			var water_sprite = food_scene.instantiate()
			water_sprite.initialize_water(water)
			food_container.add_child(water_sprite)
			active_water[water_id] = water_sprite
	
	# Limpiar agua que ya no existe
	for water_id in active_water.keys():
		if not current_water.has(water_id):
			active_water[water_id].queue_free()
			active_water.erase(water_id)

func _update_stats(data):
	if not stats_label:
		return
	
	var text = "ECOSISTEMA EN VIVO\n\n"
	text += "Paso: %d\n" % data.get("step", 0)
	text += "Episodio: %d\n" % data.get("episode", 0)
	text += "Agentes vivos: %d/%d\n" % [data.get("alive_count", 0), data.get("total_agents", 0)]
	text += "Vegetación: %d\n" % (data.vegetation.size() if data.has("vegetation") else 0)
	text += "Agua: %d" % (data.water_sources.size() if data.has("water_sources") else 0)
	
	stats_label.text = text
